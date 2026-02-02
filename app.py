import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import html as html_mod
import os
import math
import numpy as np
import duckdb
from scipy.stats import percentileofscore

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("TRACKMAN_CSV_ROOT", os.path.join(_APP_DIR, "v3"))
PARQUET_FIXED_PATH = os.path.join(_APP_DIR, "all_trackman_fixed.parquet")
PARQUET_PATH = PARQUET_FIXED_PATH if os.path.exists(PARQUET_FIXED_PATH) else os.path.join(_APP_DIR, "all_trackman.parquet")
DAVIDSON_TEAM_ID = "DAV_WIL"

ROSTER_2026 = {
    "Higgins, Justin", "Diaz, Fredy", "Vokal, Jacob", "Vannoy, Matthew",
    "Rice, Aidan", "Thomas, Gavin", "Collins, Cooper", "McCullough, Will",
    "Manjooran, Matthew", "Berman, Connor", "Daly, Jameson", "Daly, Jamie",
    "Lietz, Forrest", "Champey, Brycen", "Furr, Keely", "Loughlin, Theo",
    "Laughlin, Theo", "Cavanaugh, Cooper", "Edwards, Scotty", "Ludwig, Landon",
    "Banks, Will", "Hall, Edward", "Hall, Ed", "Torreso, Anthony", "Brooks, Will",
    "Hoyt, Ivan", "Fritch, Brendan", "Papciak, Will", "Wille, Tyler",
    "Smith, Daniel", "Jimenez, Ethan", "Jones, Parker", "Marenghi, Will",
    "Hultquist, Henry", "Whelan, Thomas", "Pyne, Garrett", "Taggart, Carson",
    "Howard, Jed", "Perkins, Wilson", "Hamilton, Matt", "Hamilton, Matthew",
    "Yochum, Simon", "Suarez, Jake", "Ban, Jason", "Katz, Adam",
}

JERSEY = {
    "Higgins, Justin": 1, "Diaz, Fredy": 2, "Vokal, Jacob": 4, "Vannoy, Matthew": 5,
    "Rice, Aidan": 6, "Thomas, Gavin": 7, "Collins, Cooper": 8, "McCullough, Will": 9,
    "Manjooran, Matthew": 10, "Berman, Connor": 11, "Daly, Jamie": 12, "Daly, Jameson": 12,
    "Lietz, Forrest": 13, "Champey, Brycen": 14, "Furr, Keely": 15,
    "Loughlin, Theo": 16, "Laughlin, Theo": 16, "Cavanaugh, Cooper": 17,
    "Edwards, Scotty": 18, "Ludwig, Landon": 19, "Banks, Will": 20,
    "Hall, Ed": 21, "Hall, Edward": 21, "Torreso, Anthony": 22, "Brooks, Will": 23,
    "Hoyt, Ivan": 24, "Fritch, Brendan": 25, "Papciak, Will": 26, "Wille, Tyler": 27,
    "Smith, Daniel": 28, "Jimenez, Ethan": 29, "Jones, Parker": 31, "Marenghi, Will": 32,
    "Hultquist, Henry": 33, "Whelan, Thomas": 34, "Pyne, Garrett": 35, "Taggart, Carson": 36,
    "Howard, Jed": 38, "Perkins, Wilson": 39, "Hamilton, Matt": 40, "Hamilton, Matthew": 40,
    "Yochum, Simon": 41, "Suarez, Jake": 42, "Ban, Jason": 43, "Katz, Adam": 44,
}

POSITION = {
    "Higgins, Justin": "INF", "Diaz, Fredy": "INF", "Vokal, Jacob": "RHP/INF",
    "Vannoy, Matthew": "C/INF", "Rice, Aidan": "INF/OF", "Thomas, Gavin": "OF",
    "Collins, Cooper": "INF", "McCullough, Will": "OF", "Manjooran, Matthew": "3B/OF",
    "Berman, Connor": "INF/OF", "Daly, Jamie": "OF", "Daly, Jameson": "OF",
    "Lietz, Forrest": "LHP/OF", "Champey, Brycen": "RHP", "Furr, Keely": "RHP",
    "Loughlin, Theo": "1B", "Laughlin, Theo": "1B", "Cavanaugh, Cooper": "LHP",
    "Edwards, Scotty": "OF", "Ludwig, Landon": "RHP", "Banks, Will": "RHP",
    "Hall, Ed": "RHP", "Hall, Edward": "RHP", "Torreso, Anthony": "C",
    "Brooks, Will": "C/INF", "Hoyt, Ivan": "RHP/INF", "Fritch, Brendan": "C/OF",
    "Papciak, Will": "RHP", "Wille, Tyler": "LHP", "Smith, Daniel": "RHP",
    "Jimenez, Ethan": "1B/RHP", "Jones, Parker": "RHP", "Marenghi, Will": "RHP",
    "Hultquist, Henry": "RHP", "Whelan, Thomas": "RHP", "Pyne, Garrett": "RHP",
    "Taggart, Carson": "LHP", "Howard, Jed": "OF/LHP", "Perkins, Wilson": "RHP",
    "Hamilton, Matt": "RHP", "Hamilton, Matthew": "RHP", "Yochum, Simon": "RHP",
    "Suarez, Jake": "C", "Ban, Jason": "LHP", "Katz, Adam": "Util",
}

NAME_MAP = {
    "Laughlin, Theo": "Loughlin, Theo",
    "Laughlin , Theo": "Loughlin, Theo",
    "Daly, Jameson": "Daly, Jamie",
    "Hall, Edward": "Hall, Ed",
    "Hamilton, Matthew": "Hamilton, Matt",
    "Edwards, Scott": "Edwards, Scotty",
    "Edwards , Scott": "Edwards, Scotty",
    "Edwards , Scotty": "Edwards, Scotty",
    "Lietz, Foresst": "Lietz, Forrest",
    "McCoullough, Will": "McCullough, Will",
}

PITCH_COLORS = {
    "Fastball": "#d22d49", "Sinker": "#fe6100", "Cutter": "#933f8e",
    "Slider": "#f7c631", "Curveball": "#00d1ed", "Changeup": "#1dbe3a",
    "Splitter": "#c99b6e", "Knuckle Curve": "#2d7fc1", "Sweeper": "#dbab00",
    "Other": "#aaaaaa",
}

# Normalize pitch type names — merge synonyms, drop junk
PITCH_TYPE_MAP = {
    "FourSeamFastBall": "Fastball",
    "OneSeamFastBall": "Sinker",
    "TwoSeamFastBall": "Sinker",
    "ChangeUp": "Changeup",
    "Knuckleball": "Other",
    "Undefined": "Other",
}
PITCH_TYPES_TO_DROP = {"Other", "Undefined"}

# ── TrueMedia → Trackman pitch type mapping ──
TM_PITCH_PCT_COLS = {
    "4Seam%": "Fastball", "Sink2Seam%": "Sinker", "Cutter%": "Cutter",
    "Slider%": "Slider", "Curve%": "Curveball", "Change%": "Changeup",
    "Split%": "Splitter", "Sweeper%": "Sweeper",
}

# ── Strike zone constants ────────────────────
ZONE_SIDE = 0.83          # feet from center (plate half-width 0.708 ft + ball radius ~0.12 ft)
ZONE_HEIGHT_BOT = 1.5     # default bottom (ft)
ZONE_HEIGHT_TOP = 3.5     # default top (ft)
MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE = 20


def safe_mode(series, default=""):
    """Return the mode of a Series, or *default* if no mode exists."""
    m = series.mode()
    return m.iloc[0] if len(m) > 0 else default


def is_barrel(ev, la):
    """Statcast barrel definition: EV >= 98 with LA range that widens as EV rises.

    At 98 mph the sweet spot is 26-30 deg.  For every 1 mph above 98 the
    acceptable LA range expands by 2-3 deg on each side, capped at 8-50 deg.
    """
    if pd.isna(ev) or pd.isna(la):
        return False
    if ev < 98:
        return False
    # At exactly 98 mph the window is 26-30
    la_min = max(26 - 2 * (ev - 98), 8)
    la_max = min(30 + 3 * (ev - 98), 50)
    return la_min <= la <= la_max


def is_barrel_mask(df):
    """Vectorised Statcast barrel mask for a DataFrame with ExitSpeed & Angle."""
    ev = pd.to_numeric(df["ExitSpeed"], errors="coerce")
    la = pd.to_numeric(df["Angle"], errors="coerce")
    la_min = (26 - 2 * (ev - 98)).clip(lower=8)
    la_max = (30 + 3 * (ev - 98)).clip(upper=50)
    return (ev >= 98) & (la >= la_min) & (la <= la_max)


@st.cache_data(show_spinner=False)
def _build_batter_zones(data):
    """Build per-batter strike zone boundaries from called-strike distributions.

    Returns dict  batter_name -> (zone_bot, zone_top).
    Falls back to fixed zone (1.5-3.5) when fewer than MIN samples exist.
    """
    zones = {}
    called = data[data["PitchCall"] == "StrikeCalled"].dropna(subset=["PlateLocHeight"])
    if called.empty:
        return zones
    for batter, grp in called.groupby("Batter"):
        if len(grp) >= MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE:
            zones[batter] = (
                round(grp["PlateLocHeight"].quantile(0.05), 3),
                round(grp["PlateLocHeight"].quantile(0.95), 3),
            )
    return zones


def in_zone_mask(df, batter_zones=None, batter_col="Batter"):
    """Per-pitch boolean mask: True if pitch is inside the batter's strike zone.

    Uses adaptive per-batter zone height when available, otherwise fixed defaults.
    Width always uses the fixed ±ZONE_SIDE (plate width doesn't change by batter).
    """
    side_ok = df["PlateLocSide"].abs() <= ZONE_SIDE
    if batter_zones and batter_col in df.columns:
        bot = df[batter_col].map(lambda b: batter_zones.get(b, (ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP))[0])
        top = df[batter_col].map(lambda b: batter_zones.get(b, (ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP))[1])
        height_ok = (df["PlateLocHeight"] >= bot) & (df["PlateLocHeight"] <= top)
    else:
        height_ok = df["PlateLocHeight"].between(ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP)
    return (side_ok & height_ok).fillna(False)


def normalize_pitch_types(df):
    """Normalize pitch type names and null out junk/undefined (instead of
    dropping rows, so KorBB outcome data on those pitches is preserved)."""
    if "TaggedPitchType" not in df.columns:
        return df
    df = df.copy()
    df["TaggedPitchType"] = df["TaggedPitchType"].replace(PITCH_TYPE_MAP)
    df.loc[df["TaggedPitchType"].isin(PITCH_TYPES_TO_DROP), "TaggedPitchType"] = np.nan
    return df


def filter_minor_pitches(df, min_pct=3.0):
    """Remove pitch types that make up less than min_pct% of a pitcher's arsenal."""
    if df.empty or "TaggedPitchType" not in df.columns:
        return df
    total = len(df)
    counts = df["TaggedPitchType"].value_counts()
    keep = counts[counts / total * 100 >= min_pct].index
    return df[df["TaggedPitchType"].isin(keep)]

st.set_page_config(page_title="Davidson Baseball Analytics", layout="wide", page_icon="⚾")
pio.templates.default = "plotly_white"

# ──────────────────────────────────────────────
# GLOBAL CSS - Savant-inspired clean white theme
# ──────────────────────────────────────────────
st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Force light backgrounds everywhere */
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main,
    .stApp { background-color: #f8f9fa !important; }

    /* Sidebar - black background, ALL red text */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] > div > div,
    [data-testid="stSidebar"] section,
    [data-testid="stSidebar"] section > div { background-color: #000000 !important; }

    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] a,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] button,
    section[data-testid="stSidebar"] *,
    section[data-testid="stSidebar"] label p,
    section[data-testid="stSidebar"] label span,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
        color: #cc0000 !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stSidebar"] label:hover,
    [data-testid="stSidebar"] label:hover p,
    [data-testid="stSidebar"] label:hover span { color: #ff3333 !important; }
    [data-testid="stSidebar"] hr { border-color: #cc0000 !important; opacity: 0.3 !important; }
    /* Sidebar brand header — force red with high specificity */
    .sidebar-brand-title,
    [data-testid="stSidebar"] .sidebar-brand-title,
    section[data-testid="stSidebar"] .sidebar-brand-title,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] .sidebar-brand-title { color: #cc0000 !important; }
    .sidebar-brand-sub,
    [data-testid="stSidebar"] .sidebar-brand-sub,
    section[data-testid="stSidebar"] .sidebar-brand-sub,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] .sidebar-brand-sub { color: #cc0000 !important; opacity: 0.6; }
    /* Radio buttons and icons */
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] button svg,
    [data-testid="stSidebar"] svg,
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarCollapseButton"] svg {
        color: #cc0000 !important;
        fill: #cc0000 !important;
    }

    /* Expander fix */
    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
        margin-top: 8px !important;
    }
    [data-testid="stExpander"] summary {
        color: #1a1a2e !important;
        font-weight: 600 !important;
    }
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p {
        color: #1a1a2e !important;
    }

    /* Force dark text everywhere in main content */
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] h4,
    [data-testid="stAppViewContainer"] h5,
    [data-testid="stAppViewContainer"] h6,
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] span,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] div,
    [data-testid="stAppViewContainer"] .stMarkdown,
    [data-testid="stAppViewContainer"] .stCaption,
    [data-testid="stAppViewContainer"] [data-testid="stMetricValue"],
    [data-testid="stAppViewContainer"] [data-testid="stMetricLabel"],
    [data-testid="stAppViewContainer"] .stSelectbox label,
    [data-testid="stAppViewContainer"] .stMultiSelect label,
    [data-testid="stAppViewContainer"] .stRadio label,
    [data-testid="stAppViewContainer"] .stTabs [data-baseweb="tab"] {
        color: #1a1a2e !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Player header on dark background — override main content dark text */
    .player-header-dark .ph-name { color: #cc0000 !important; }
    .player-header-dark .ph-jersey { color: #cc0000 !important; }
    .player-header-dark .ph-detail { color: #ff6666 !important; }
    .player-header-dark .ph-stat { color: #cc9999 !important; }

    /* Dataframes */
    [data-testid="stAppViewContainer"] .stDataFrame,
    [data-testid="stAppViewContainer"] [data-testid="stDataFrame"] * {
        color: #1a1a2e !important;
    }

    /* Chart containers */
    [data-testid="stAppViewContainer"] .stPlotlyChart {
        background-color: #ffffff !important;
        border-radius: 8px;
    }

    .block-container { padding-top: 1rem; max-width: 1400px; }

    /* Section headers */
    .section-header {
        font-size: 14px !important;
        font-weight: 700 !important;
        color: #1a1a2e !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: 2px solid #cc0000;
        padding-bottom: 6px;
        margin-bottom: 12px !important;
        margin-top: 8px !important;
    }
</style>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to database...")
def get_duckdb_con():
    """Return a DuckDB connection with a VIEW over the full parquet file."""
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(f"Parquet file not found: {PARQUET_PATH}")
    con = duckdb.connect(database=':memory:')
    # Normalize TaggedPitchType at the DB layer so all SQL queries are consistent.
    con.execute(
        f"""
        CREATE VIEW trackman AS
        SELECT
            * EXCLUDE (TaggedPitchType),
            CASE
                WHEN TaggedPitchType IN ('Undefined','Other','Knuckleball') THEN NULL
                WHEN TaggedPitchType IN ('FourSeamFastBall','OneSeamFastBall') THEN 'Fastball'
                WHEN TaggedPitchType = 'TwoSeamFastBall' THEN 'Sinker'
                WHEN TaggedPitchType = 'ChangeUp' THEN 'Changeup'
                ELSE TaggedPitchType
            END AS TaggedPitchType
        FROM read_parquet('{PARQUET_PATH}')
        WHERE PitchCall IS NULL OR PitchCall != 'Undefined'
        """
    )
    return con


def query_population(sql):
    """Run an ad-hoc SQL query against the full D1 parquet via DuckDB."""
    return get_duckdb_con().execute(sql).fetchdf()


@st.cache_data(show_spinner="Loading Davidson data...")
def load_davidson_data():
    """Load only Davidson rows from parquet into pandas (~300k rows)."""
    if not os.path.exists(PARQUET_PATH):
        return pd.DataFrame()
    sql = f"""
        SELECT * FROM read_parquet('{PARQUET_PATH}')
        WHERE (PitcherTeam = '{DAVIDSON_TEAM_ID}' OR BatterTeam = '{DAVIDSON_TEAM_ID}')
          AND (PitchCall IS NULL OR PitchCall != 'Undefined')
    """
    data = duckdb.query(sql).fetchdf()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    if "Season" in data.columns:
        data["Season"] = pd.to_numeric(data["Season"], errors="coerce").astype("Int64")
    for col in ["Pitcher", "Batter"]:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip().replace(NAME_MAP)
    data = normalize_pitch_types(data)
    return data


@st.cache_data(show_spinner=False)
def get_all_seasons():
    """Return sorted list of all seasons in the full D1 database."""
    df = query_population("SELECT DISTINCT Season FROM trackman WHERE Season IS NOT NULL AND Season > 0 ORDER BY Season")
    return sorted(df["Season"].dropna().astype(int).tolist())


@st.cache_data(show_spinner=False)
def get_sidebar_stats():
    """Return sidebar aggregate counts from full D1 database via DuckDB."""
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


def _is_position_player(name):
    """True if player is a position player (not a pure pitcher)."""
    pos = POSITION.get(name, "")
    return pos not in ("RHP", "LHP")


def filter_davidson(data, role="pitcher"):
    if role == "pitcher":
        return data[(data["PitcherTeam"] == DAVIDSON_TEAM_ID) & (data["Pitcher"].isin(ROSTER_2026))].copy()
    else:
        return data[(data["BatterTeam"] == DAVIDSON_TEAM_ID) & (data["Batter"].isin(ROSTER_2026))].copy()


# ──────────────────────────────────────────────
# PERCENTILE ENGINE
# ──────────────────────────────────────────────
SWING_CALLS = ["StrikeSwinging", "FoulBall", "FoulBallNotFieldable", "FoulBallFieldable", "InPlay"]
CONTACT_CALLS = ["FoulBall", "FoulBallNotFieldable", "FoulBallFieldable", "InPlay"]


# ──────────────────────────────────────────────
# TRUEMEDIA DATA LOADER
# ──────────────────────────────────────────────
def _pct_to_float(s):
    """Convert '42.8%' → 42.8 (float). Leaves non-% values unchanged."""
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


# ──────────────────────────────────────────────
# DuckDB-BACKED POPULATION STATS (replaces old pandas-only versions)
# ──────────────────────────────────────────────
_SWING_CALLS_SQL = "('StrikeSwinging','FoulBall','FoulBallNotFieldable','FoulBallFieldable','InPlay')"
_CONTACT_CALLS_SQL = "('FoulBall','FoulBallNotFieldable','FoulBallFieldable','InPlay')"
_IZ_COND = "ABS(PlateLocSide) <= 0.83 AND PlateLocHeight BETWEEN 1.5 AND 3.5"
_HAS_LOC = "PlateLocSide IS NOT NULL AND PlateLocHeight IS NOT NULL"
_OZ_COND = f"NOT ({_IZ_COND}) AND {_HAS_LOC}"

# Build SQL CASE expression for name normalization (matches NAME_MAP applied in load_davidson_data)
def _name_sql(col):
    """Return SQL CASE expression that normalizes player names to match NAME_MAP."""
    def _esc(s):
        return s.replace("'", "''")
    parts = " ".join(f"WHEN TRIM({col}) = '{_esc(old)}' THEN '{_esc(new)}'" for old, new in NAME_MAP.items())
    return f"CASE {parts} ELSE TRIM({col}) END"


@st.cache_data(show_spinner="Computing batter rankings...")
def compute_batter_stats_pop(season_filter=None, _version=5):
    """Compute batter stats for all D1 batters via DuckDB. Adaptive per-batter zone."""
    season_clause = ""
    if season_filter:
        seasons_in = ",".join(str(int(s)) for s in season_filter)
        season_clause = f"AND Season IN ({seasons_in})"

    _bnorm = _name_sql("Batter")
    sql = f"""
    WITH batter_zones AS (
        SELECT {_bnorm} AS batter_name,
            CASE WHEN COUNT(*) >= {MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE}
                 THEN ROUND(PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY PlateLocHeight), 3)
                 ELSE {ZONE_HEIGHT_BOT} END AS zone_bot,
            CASE WHEN COUNT(*) >= {MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE}
                 THEN ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY PlateLocHeight), 3)
                 ELSE {ZONE_HEIGHT_TOP} END AS zone_top
        FROM trackman
        WHERE PitchCall = 'StrikeCalled'
          AND PlateLocHeight IS NOT NULL
          AND Batter IS NOT NULL {season_clause}
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
        FROM trackman r
        LEFT JOIN batter_zones bz ON {_bnorm} = bz.batter_name
        WHERE Batter IS NOT NULL {season_clause}
    ),
    pitch_agg AS (
        SELECT
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
        HAVING PA > 50
    )
    SELECT * FROM pitch_agg
    """
    agg = query_population(sql)
    if agg.empty:
        return agg

    # Compute derived percentages — use np.where guards so zero-denominator
    # cases produce NaN (not 0%), preventing phantom 0% entries from
    # corrupting percentile distributions.
    n = agg["bbe"]
    pa = agg["PA"]
    sw = agg["swings"]
    oz = agg["oz_count"]
    iz = agg["iz_count"]
    iz_sw = agg["iz_swings"]
    oz_sw = agg["oz_swings"]

    agg["HardHitPct"] = np.where(n > 0, agg["hard_hits"] / n * 100, np.nan)
    agg["BarrelPct"] = np.where(n > 0, agg["Barrels"] / n * 100, np.nan)
    agg["BarrelPA"] = np.where(pa > 0, agg["Barrels"] / pa * 100, np.nan)
    agg["SweetSpotPct"] = np.where(n > 0, agg["sweet_spots"] / n * 100, np.nan)
    agg["WhiffPct"] = np.where(sw > 0, agg["whiffs"] / sw * 100, np.nan)
    agg["KPct"] = np.where(pa > 0, agg["ks"] / pa * 100, np.nan)
    agg["BBPct"] = np.where(pa > 0, agg["bbs"] / pa * 100, np.nan)
    agg["ChasePct"] = np.where(oz > 0, agg["oz_swings"] / oz * 100, np.nan)
    agg["ChaseContact"] = np.where(oz_sw > 0, agg["oz_contacts"] / oz_sw * 100, np.nan)
    agg["ZoneSwingPct"] = np.where(iz > 0, iz_sw / iz * 100, np.nan)
    agg["ZoneContactPct"] = np.where(iz_sw > 0, agg["iz_contacts"] / iz_sw * 100, np.nan)
    agg["ZonePct"] = np.where(agg["n_loc"] > 0, agg["iz_count"] / agg["n_loc"] * 100, np.nan)
    agg["SwingPct"] = np.where(agg["n_pitches"] > 0, sw / agg["n_pitches"] * 100, np.nan)
    agg["GBPct"] = np.where(n > 0, agg["gb"] / n * 100, np.nan)
    agg["FBPct"] = np.where(n > 0, agg["fb"] / n * 100, np.nan)
    agg["LDPct"] = np.where(n > 0, agg["ld"] / n * 100, np.nan)
    agg["PUPct"] = np.where(n > 0, agg["pu"] / n * 100, np.nan)
    agg["AirPct"] = np.where(n > 0, (agg["fb"] + agg["ld"] + agg["pu"]) / n * 100, np.nan)
    agg["BBE"] = agg["bbe"]

    keep = ["Batter", "BatterTeam", "PA", "BBE", "AvgEV", "MaxEV", "HardHitPct",
            "Barrels", "BarrelPct", "BarrelPA", "SweetSpotPct", "AvgLA", "AvgDist",
            "WhiffPct", "KPct", "BBPct", "ChasePct", "ChaseContact",
            "ZoneSwingPct", "ZoneContactPct", "ZonePct", "SwingPct",
            "GBPct", "FBPct", "LDPct", "PUPct", "AirPct"]
    return agg[[c for c in keep if c in agg.columns]]


@st.cache_data(show_spinner="Computing pitcher rankings...")
def compute_pitcher_stats_pop(season_filter=None, _version=5):
    """Compute pitcher stats for all D1 pitchers via DuckDB. Adaptive per-batter zone."""
    season_clause = ""
    if season_filter:
        seasons_in = ",".join(str(int(s)) for s in season_filter)
        season_clause = f"AND Season IN ({seasons_in})"

    _pnorm = _name_sql("Pitcher")
    _bnorm_p = _name_sql("Batter")
    sql = f"""
    WITH batter_zones AS (
        SELECT {_bnorm_p} AS batter_name,
            CASE WHEN COUNT(*) >= {MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE}
                 THEN ROUND(PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY PlateLocHeight), 3)
                 ELSE {ZONE_HEIGHT_BOT} END AS zone_bot,
            CASE WHEN COUNT(*) >= {MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE}
                 THEN ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY PlateLocHeight), 3)
                 ELSE {ZONE_HEIGHT_TOP} END AS zone_top
        FROM trackman
        WHERE PitchCall = 'StrikeCalled'
          AND PlateLocHeight IS NOT NULL
          AND Batter IS NOT NULL {season_clause}
        GROUP BY {_bnorm_p}
    ),
    raw AS (
        SELECT
            {_pnorm} AS Pitcher, PitcherTeam,
            PitchCall, ExitSpeed, Angle, TaggedHitType, TaggedPitchType,
            PlateLocSide, PlateLocHeight, RelSpeed, SpinRate, Extension, KorBB,
            GameID || '_' || Inning || '_' || PAofInning || '_' || {_bnorm_p} AS pa_id,
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
        FROM trackman r
        LEFT JOIN batter_zones bz ON {_bnorm_p} = bz.batter_name
        WHERE Pitcher IS NOT NULL {season_clause}
    ),
    pitch_agg AS (
        SELECT
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
    )
    SELECT * FROM pitch_agg
    """
    agg = query_population(sql)
    if agg.empty:
        return agg

    pa = agg["PA"]
    sw = agg["swings"]
    oz = agg["oz_count"]
    iz_sw = agg["iz_swings"]
    n_bat = agg["bbe"]
    n_ip = agg["n_ip"]

    agg["WhiffPct"] = np.where(sw > 0, agg["whiffs"] / sw * 100, np.nan)
    agg["KPct"] = np.where(pa > 0, agg["ks"] / pa * 100, np.nan)
    agg["BBPct"] = np.where(pa > 0, agg["bbs"] / pa * 100, np.nan)
    agg["ZonePct"] = np.where(agg["n_loc"] > 0, agg["iz_count"] / agg["n_loc"] * 100, np.nan)
    agg["ChasePct"] = np.where(oz > 0, agg["oz_swings"] / oz * 100, np.nan)
    agg["ZoneContactPct"] = np.where(iz_sw > 0, agg["iz_contacts"] / iz_sw * 100, np.nan)
    agg["HardHitAgainst"] = np.where(n_bat > 0, agg["hard_hits"] / n_bat * 100, np.nan)
    agg["BarrelPctAgainst"] = np.where(n_bat > 0, agg["n_barrels"] / n_bat * 100, np.nan)
    agg["GBPct"] = np.where(n_ip > 0, agg["gb"] / n_ip * 100, np.nan)
    agg["SwingPct"] = np.where(agg["Pitches"] > 0, sw / agg["Pitches"] * 100, np.nan)

    keep = ["Pitcher", "PitcherTeam", "Pitches", "PA", "AvgFBVelo", "MaxFBVelo",
            "AvgSpin", "Extension", "WhiffPct", "KPct", "BBPct", "ZonePct",
            "ChasePct", "ZoneContactPct", "AvgEVAgainst", "HardHitAgainst",
            "BarrelPctAgainst", "GBPct", "SwingPct"]
    return agg[[c for c in keep if c in agg.columns]]


@st.cache_data(show_spinner=False)
def compute_stuff_baselines():
    """Pre-compute per-pitch-type mean/std for Stuff+ via DuckDB.
    Returns dict: {pitch_type: {col: (mean, std)}} and fb_velo_by_pitcher dict.
    """
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
    df = query_population(sql)
    baseline_stats = {}
    for _, row in df.iterrows():
        pt = row["TaggedPitchType"]
        stats = {}
        for col in base_cols:
            m, s = row.get(f"{col}_mean"), row.get(f"{col}_std")
            if pd.notna(m) and pd.notna(s):
                stats[col] = (float(m), float(s))
        baseline_stats[pt] = stats

    # Per-pitcher fastball velo for VeloDiff
    fb_sql = """
        SELECT Pitcher, AVG(RelSpeed) AS fb_velo
        FROM trackman
        WHERE TaggedPitchType IN ('Fastball','Sinker','Cutter','FourSeamFastBall','TwoSeamFastBall','OneSeamFastBall')
          AND RelSpeed IS NOT NULL
        GROUP BY Pitcher
    """
    fb_df = query_population(fb_sql)
    fb_velo_by_pitcher = dict(zip(fb_df["Pitcher"], fb_df["fb_velo"]))

    # VeloDiff baselines for changeups/splitters
    velo_diff_stats = {}
    for pt in ["Changeup", "Splitter"]:
        # Map raw types to normalized for matching
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
        try:
            vd_df = query_population(vd_sql)
            if len(vd_df) > 2:
                vd = vd_df["fb_velo"] - vd_df["RelSpeed"]
                velo_diff_stats[pt] = (float(vd.mean()), float(vd.std()))
        except Exception:
            pass

    return {"baseline_stats": baseline_stats, "fb_velo_by_pitcher": fb_velo_by_pitcher,
            "velo_diff_stats": velo_diff_stats}


@st.cache_data(show_spinner="Building tunnel population database...")
def build_tunnel_population_pop():
    """Build tunnel population from DuckDB: per-pitcher per-pitch-type aggregates,
    then Python Euler physics for tunnel scores."""
    sql = """
    WITH pitcher_elig AS (
        SELECT Pitcher
        FROM trackman
        WHERE TaggedPitchType NOT IN ('Other','Undefined')
        GROUP BY Pitcher
        HAVING COUNT(*) >= 50 AND COUNT(DISTINCT TaggedPitchType) >= 2
    ),
    pt_agg AS (
        SELECT
            t.Pitcher, t.TaggedPitchType,
            AVG(t.RelHeight) AS rel_h, AVG(t.RelSide) AS rel_s,
            STDDEV(t.RelHeight) AS rel_h_std, STDDEV(t.RelSide) AS rel_s_std,
            AVG(t.PlateLocHeight) AS loc_h, AVG(t.PlateLocSide) AS loc_s,
            AVG(t.RelSpeed) AS velo, COUNT(t.RelSpeed) AS cnt,
            AVG(t.InducedVertBreak) AS ivb, AVG(t.HorzBreak) AS hb,
            AVG(t.Extension) AS ext
        FROM trackman t
        JOIN pitcher_elig pe ON t.Pitcher = pe.Pitcher
        WHERE t.TaggedPitchType NOT IN ('Other','Undefined')
        GROUP BY t.Pitcher, t.TaggedPitchType
        HAVING COUNT(t.RelSpeed) >= 10
    )
    SELECT * FROM pt_agg
    """
    agg_df = query_population(sql)
    if agg_df.empty:
        return {}

    agg_df["rel_h_std"] = agg_df["rel_h_std"].fillna(0)
    agg_df["rel_s_std"] = agg_df["rel_s_std"].fillna(0)
    agg_df["ext"] = agg_df["ext"].fillna(6.0)

    # Physics constants
    MOUND_DIST = 60.5
    GRAVITY = 32.17
    COMMIT_TIME = 0.280
    N_STEPS = 20

    # AGGREGATE-level benchmarks for population builder (uses mean stats per pitch type).
    # Aggregate commit seps are ~2x smaller than PBP because averaging removes variation.
    # Format: (p10, p25, p50, p75, p90, mean, std)
    PAIR_BENCHMARKS = {
        'Changeup/Curveball': (2.2, 3.5, 5.2, 7.1, 9.3, 5.6, 3.1),
        'Changeup/Cutter': (1.4, 2.2, 3.4, 4.9, 6.7, 3.9, 2.7),
        'Changeup/Fastball': (1.8, 2.7, 4.0, 5.2, 6.6, 4.2, 2.1),
        'Changeup/Sinker': (1.6, 2.4, 3.6, 5.0, 6.7, 4.0, 2.9),
        'Changeup/Slider': (1.4, 2.2, 3.4, 4.8, 6.5, 3.8, 2.4),
        'Changeup/Splitter': (1.0, 1.7, 2.6, 4.2, 6.1, 3.3, 2.8),
        'Changeup/Sweeper': (1.7, 2.7, 4.4, 6.3, 8.6, 4.8, 2.7),
        'Curveball/Cutter': (1.6, 2.5, 4.0, 5.8, 7.7, 4.4, 2.9),
        'Curveball/Fastball': (1.5, 2.4, 3.7, 5.3, 7.1, 4.1, 2.6),
        'Curveball/Sinker': (1.4, 2.5, 4.0, 6.0, 8.8, 4.9, 4.0),
        'Curveball/Slider': (1.2, 2.0, 3.3, 5.0, 7.1, 4.0, 3.1),
        'Curveball/Splitter': (2.1, 3.1, 4.6, 6.5, 9.1, 5.1, 2.8),
        'Curveball/Sweeper': (1.1, 2.6, 3.8, 5.8, 8.0, 5.0, 5.0),
        'Cutter/Fastball': (0.9, 1.5, 2.5, 3.8, 5.3, 3.0, 2.3),
        'Cutter/Sinker': (1.0, 1.7, 2.7, 4.2, 6.3, 3.3, 2.9),
        'Cutter/Slider': (0.9, 1.4, 2.4, 3.7, 5.4, 2.9, 2.4),
        'Cutter/Splitter': (1.0, 1.7, 3.0, 5.0, 7.2, 3.6, 2.5),
        'Cutter/Sweeper': (1.6, 2.6, 4.1, 5.8, 7.6, 4.3, 2.2),
        'Fastball/Sinker': (0.8, 1.4, 2.3, 3.7, 5.6, 3.0, 2.8),
        'Fastball/Slider': (1.1, 1.8, 2.9, 4.1, 5.6, 3.2, 2.0),
        'Fastball/Splitter': (1.2, 2.0, 3.4, 4.8, 6.7, 3.7, 2.3),
        'Fastball/Sweeper': (1.7, 2.9, 4.7, 7.1, 8.9, 5.2, 3.2),
        'Sinker/Slider': (1.1, 1.9, 3.0, 4.4, 6.1, 3.5, 2.7),
        'Sinker/Splitter': (1.1, 2.1, 3.4, 5.2, 7.7, 4.0, 2.8),
        'Sinker/Sweeper': (1.4, 2.5, 4.0, 5.9, 8.5, 4.5, 2.4),
        'Slider/Splitter': (1.1, 1.9, 3.2, 4.7, 7.0, 3.7, 2.7),
        'Slider/Sweeper': (1.1, 1.8, 2.8, 4.5, 6.5, 3.5, 2.4),
    }
    DEFAULT_BENCHMARK = (1.2, 2.1, 3.3, 4.9, 6.8, 3.8, 2.7)

    def _euler_traj(rel_h, rel_s, loc_h, loc_s, ivb_val, hb_val, velo_mph, ext_val):
        ext_val = ext_val if not pd.isna(ext_val) else 6.0
        actual_dist = MOUND_DIST - ext_val
        velo_fps = velo_mph * 5280.0 / 3600.0
        if velo_fps < 50:
            velo_fps = 50.0
        t_total = actual_dist / velo_fps
        dt = t_total / N_STEPS
        ivb_ft = ivb_val / 12.0
        hb_ft = hb_val / 12.0
        a_ivb = 2.0 * ivb_ft / (t_total ** 2) if t_total > 0 else 0
        a_hb = 2.0 * hb_ft / (t_total ** 2) if t_total > 0 else 0
        T = t_total
        vy0 = (loc_h - rel_h + 0.5 * GRAVITY * T**2 - 0.5 * a_ivb * T**2) / T if T > 0 else 0
        vx0 = (loc_s - rel_s - 0.5 * a_hb * T**2) / T if T > 0 else 0
        path = [(rel_s, rel_h, 0.0)]
        x, y = rel_s, rel_h
        vy, vx = vy0, vx0
        t = 0.0
        for _ in range(N_STEPS):
            vy += (-GRAVITY + a_ivb) * dt
            vx += a_hb * dt
            y += vy * dt
            x += vx * dt
            t += dt
            path.append((x, y, t))
        return path, t_total

    def _tunnel_from_agg_row(row):
        ivb = row["ivb"] if pd.notna(row["ivb"]) else 0.0
        hb = row["hb"] if pd.notna(row["hb"]) else 0.0
        return _euler_traj(row["rel_h"], row["rel_s"], row["loc_h"], row["loc_s"],
                           ivb, hb, row["velo"], row["ext"])

    def _pos_at_time_pop(path, t_total, target_t):
        """Interpolate (x, y) at a specific time from an Euler path."""
        if target_t <= 0:
            return path[0][0], path[0][1]
        if target_t >= t_total:
            return path[-1][0], path[-1][1]
        dt = t_total / N_STEPS
        idx_f = target_t / dt
        idx = int(idx_f)
        frac = idx_f - idx
        if idx >= len(path) - 1:
            return path[-1][0], path[-1][1]
        x = path[idx][0] + frac * (path[idx + 1][0] - path[idx][0])
        y = path[idx][1] + frac * (path[idx + 1][1] - path[idx][1])
        return x, y

    # Group by pitcher, compute tunnel scores using same formula as _compute_tunnel_score
    pop = {}
    for pitcher, p_agg in agg_df.groupby("Pitcher"):
        if len(p_agg) < 2:
            continue
        p_agg_idx = p_agg.set_index("TaggedPitchType")
        # Compute trajectories for each pitch type
        trajs = {}
        for pt, row in p_agg_idx.iterrows():
            try:
                path, t_total = _tunnel_from_agg_row(row)
                trajs[pt] = {"path": path, "t_total": t_total, "row": row}
            except Exception:
                continue
        if len(trajs) < 2:
            continue

        # Compute pair scores — mirrors _compute_tunnel_score scoring exactly
        pts = list(trajs.keys())
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                a_name, b_name = pts[i], pts[j]
                ta, tb = trajs[a_name], trajs[b_name]
                a_row, b_row = ta["row"], tb["row"]

                # Commit separation (speed-adjusted, 280ms before plate)
                commit_t_a = max(0, ta["t_total"] - COMMIT_TIME)
                commit_t_b = max(0, tb["t_total"] - COMMIT_TIME)
                cax, cay = _pos_at_time_pop(ta["path"], ta["t_total"], commit_t_a)
                cbx, cby = _pos_at_time_pop(tb["path"], tb["t_total"], commit_t_b)
                commit_sep = math.sqrt((cay - cby)**2 + (cax - cbx)**2) * 12

                # Plate separation
                pax, pay = ta["path"][-1][0], ta["path"][-1][1]
                pbx, pby = tb["path"][-1][0], tb["path"][-1][1]
                plate_sep = math.sqrt((pay - pby)**2 + (pax - pbx)**2) * 12

                # Release separation
                rel_sep = math.sqrt((a_row["rel_h"] - b_row["rel_h"])**2 +
                                    (a_row["rel_s"] - b_row["rel_s"])**2) * 12

                # Release-point variance penalty
                combined_rel_std = math.sqrt(
                    (a_row["rel_h_std"]**2 + b_row["rel_h_std"]**2) / 2 +
                    (a_row["rel_s_std"]**2 + b_row["rel_s_std"]**2) / 2
                ) * 12
                effective_rel_sep = rel_sep + 0.5 * combined_rel_std

                # Movement divergence (IVB/HB based)
                ivb_a = a_row["ivb"] if pd.notna(a_row["ivb"]) else 0
                hb_a = a_row["hb"] if pd.notna(a_row["hb"]) else 0
                ivb_b = b_row["ivb"] if pd.notna(b_row["ivb"]) else 0
                hb_b = b_row["hb"] if pd.notna(b_row["hb"]) else 0
                move_div = math.sqrt((ivb_a - ivb_b)**2 + (hb_a - hb_b)**2)

                velo_gap = abs(a_row["velo"] - b_row["velo"])

                # Pair benchmarks
                pair_key = '/'.join(sorted([a_name, b_name]))
                bm = PAIR_BENCHMARKS.get(pair_key, DEFAULT_BENCHMARK)
                bm_p10, bm_p25, bm_p50, bm_p75, bm_p90 = bm[0], bm[1], bm[2], bm[3], bm[4]

                # Commit percentile (same interpolation as _compute_tunnel_score)
                _anchors = [
                    (bm_p90, 10), (bm_p75, 25), (bm_p50, 50),
                    (bm_p25, 75), (bm_p10, 90),
                ]
                if commit_sep >= bm_p90:
                    commit_pct = max(0, 10 * (1 - (commit_sep - bm_p90) / max(bm_p90, 1)))
                elif commit_sep <= bm_p10:
                    commit_pct = min(100, 90 + 10 * (bm_p10 - commit_sep) / max(bm_p10, 1))
                else:
                    commit_pct = 50.0
                    for k in range(len(_anchors) - 1):
                        sep_hi, pct_lo = _anchors[k]
                        sep_lo, pct_hi = _anchors[k + 1]
                        if sep_lo <= commit_sep <= sep_hi:
                            frac = (sep_hi - commit_sep) / (sep_hi - sep_lo) if sep_hi != sep_lo else 0.5
                            commit_pct = pct_lo + frac * (pct_hi - pct_lo)
                            break

                plate_pct = min(100, plate_sep / 30.0 * 100)
                rel_pct = max(0, 100 - effective_rel_sep * 12)
                rel_angle_pct = 50  # no release angle data in population agg
                move_pct = min(100, move_div / 30.0 * 100)
                velo_pct = min(100, velo_gap / 15.0 * 100)

                raw_tunnel = round(
                    commit_pct * 0.55 +
                    plate_pct * 0.19 +
                    rel_pct * 0.10 +
                    rel_angle_pct * 0.08 +
                    move_pct * 0.06 +
                    velo_pct * 0.02, 2)

                pop.setdefault(pair_key, []).append(raw_tunnel)

    for k in pop:
        pop[k] = np.array(sorted(pop[k]))
    return pop


@st.cache_data(show_spinner=False)
def compute_batter_stats(data, season_filter=None):
    df = data.copy()
    if season_filter:
        df = df[df["Season"].isin(season_filter)]
    batter_zones = _build_batter_zones(df)
    _in_zone = in_zone_mask(df, batter_zones, batter_col="Batter")
    has_loc = df["PlateLocSide"].notna() & df["PlateLocHeight"].notna()
    out_zone = ~_in_zone & has_loc

    # Build unique PA identifier (robust to sampled data where PitchofPA==1 may be missing)
    _pa_id_cols = ["GameID", "Inning", "PAofInning", "Batter"]
    if all(c in df.columns for c in _pa_id_cols):
        df["_pa_id"] = (df["GameID"].astype(str) + "_" + df["Inning"].astype(str)
                        + "_" + df["PAofInning"].astype(str) + "_" + df["Batter"].astype(str))
    else:
        df["_pa_id"] = df["PitchofPA"].astype(str) + "_" + df.index.astype(str)

    # Pre-compute boolean columns for vectorized groupby
    df["_is_swing"] = df["PitchCall"].isin(SWING_CALLS)
    df["_is_whiff"] = df["PitchCall"] == "StrikeSwinging"
    df["_is_contact"] = df["PitchCall"].isin(CONTACT_CALLS)
    df["_is_inplay"] = df["PitchCall"] == "InPlay"
    df["_in_zone"] = _in_zone
    df["_out_zone"] = out_zone
    df["_has_loc"] = has_loc
    df["_iz_swing"] = _in_zone & df["_is_swing"]
    df["_oz_swing"] = out_zone & df["_is_swing"]
    df["_iz_contact"] = _in_zone & df["_is_contact"]
    df["_oz_contact"] = out_zone & df["_is_contact"]
    df["_has_ev"] = df["_is_inplay"] & df["ExitSpeed"].notna()
    df["_hard_hit"] = df["_has_ev"] & (df["ExitSpeed"] >= 95)
    if "Angle" in df.columns:
        df["_sweet_spot"] = df["_has_ev"] & df["Angle"].between(8, 32)
    else:
        df["_sweet_spot"] = False

    # Count unique PAs and deduplicated K/BB per batter
    pa_counts = df.groupby(["Batter", "BatterTeam"])["_pa_id"].nunique().reset_index(name="PA")
    k_pas = df[df["KorBB"] == "Strikeout"].groupby(["Batter", "BatterTeam"])["_pa_id"].nunique().reset_index(name="ks")
    bb_pas = df[df["KorBB"] == "Walk"].groupby(["Batter", "BatterTeam"])["_pa_id"].nunique().reset_index(name="bbs")

    # Vectorized groupby aggregation
    grp = df.groupby(["Batter", "BatterTeam"])
    agg = grp.agg(
        n_pitches=("_is_swing", "size"),
        swings=("_is_swing", "sum"),
        whiffs=("_is_whiff", "sum"),
        contacts=("_is_contact", "sum"),
        n_loc=("_has_loc", "sum"),
        iz_count=("_in_zone", "sum"),
        oz_count=("_out_zone", "sum"),
        iz_swings=("_iz_swing", "sum"),
        oz_swings=("_oz_swing", "sum"),
        iz_contacts=("_iz_contact", "sum"),
        oz_contacts=("_oz_contact", "sum"),
        bbe=("_has_ev", "sum"),
        hard_hits=("_hard_hit", "sum"),
        sweet_spots=("_sweet_spot", "sum"),
    ).reset_index()

    # Merge correct PA and K/BB counts
    agg = agg.merge(pa_counts, on=["Batter", "BatterTeam"], how="left")
    agg = agg.merge(k_pas, on=["Batter", "BatterTeam"], how="left")
    agg = agg.merge(bb_pas, on=["Batter", "BatterTeam"], how="left")
    agg["PA"] = agg["PA"].fillna(0).astype(int)
    agg["ks"] = agg["ks"].fillna(0).astype(int)
    agg["bbs"] = agg["bbs"].fillna(0).astype(int)

    # Filter to min 50 PA
    agg = agg[agg["PA"] >= 25].copy()

    # Batted ball sub-aggregations (only for in-play with EV)
    batted = df[df["_has_ev"]].copy()
    if not batted.empty:
        batted_agg = batted.groupby(["Batter", "BatterTeam"]).agg(
            AvgEV=("ExitSpeed", "mean"),
            MaxEV=("ExitSpeed", "max"),
            AvgLA=("Angle", "mean"),
            AvgDist=("Distance", "mean"),
        ).reset_index()
        # Barrel computation
        batted["_barrel"] = is_barrel_mask(batted)
        barrel_agg = batted.groupby(["Batter", "BatterTeam"])["_barrel"].sum().reset_index()
        barrel_agg.columns = ["Batter", "BatterTeam", "Barrels"]
        batted_agg = batted_agg.merge(barrel_agg, on=["Batter", "BatterTeam"], how="left")
        # Hit type counts
        for ht, col_name in [("GroundBall", "gb"), ("FlyBall", "fb"), ("LineDrive", "ld"), ("Popup", "pu")]:
            if "TaggedHitType" in batted.columns:
                ht_counts = batted[batted["TaggedHitType"] == ht].groupby(["Batter", "BatterTeam"]).size().reset_index(name=col_name)
                batted_agg = batted_agg.merge(ht_counts, on=["Batter", "BatterTeam"], how="left")
            else:
                batted_agg[col_name] = 0
        batted_agg = batted_agg.fillna({"gb": 0, "fb": 0, "ld": 0, "pu": 0, "Barrels": 0})
        agg = agg.merge(batted_agg, on=["Batter", "BatterTeam"], how="left")
    else:
        for c in ["AvgEV", "MaxEV", "AvgLA", "AvgDist", "Barrels", "gb", "fb", "ld", "pu"]:
            agg[c] = np.nan

    # Compute derived percentages
    n = agg["bbe"]
    pa = agg["PA"]
    sw = agg["swings"]
    oz = agg["oz_count"]
    iz = agg["iz_count"]
    iz_sw = agg["iz_swings"]
    oz_sw = agg["oz_swings"]

    agg["HardHitPct"] = np.where(n > 0, agg["hard_hits"] / n * 100, np.nan)
    agg["BarrelPct"] = np.where(n > 0, agg["Barrels"] / n * 100, np.nan)
    agg["BarrelPA"] = np.where(pa > 0, agg["Barrels"] / pa * 100, np.nan)
    agg["SweetSpotPct"] = np.where(n > 0, agg["sweet_spots"] / n * 100, np.nan)
    agg["WhiffPct"] = np.where(sw > 0, agg["whiffs"] / sw * 100, np.nan)
    agg["KPct"] = np.where(pa > 0, agg["ks"] / pa * 100, np.nan)
    agg["BBPct"] = np.where(pa > 0, agg["bbs"] / pa * 100, np.nan)
    agg["ChasePct"] = np.where(oz > 0, agg["oz_swings"] / oz * 100, np.nan)
    agg["ChaseContact"] = np.where(oz_sw > 0, agg["oz_contacts"] / oz_sw * 100, np.nan)
    agg["ZoneSwingPct"] = np.where(iz > 0, iz_sw / iz * 100, np.nan)
    agg["ZoneContactPct"] = np.where(iz_sw > 0, agg["iz_contacts"] / iz_sw * 100, np.nan)
    agg["ZonePct"] = np.where(agg["n_loc"] > 0, agg["iz_count"] / agg["n_loc"] * 100, np.nan)
    agg["SwingPct"] = np.where(agg["n_pitches"] > 0, sw / agg["n_pitches"] * 100, np.nan)
    agg["GBPct"] = np.where(n > 0, agg["gb"] / n * 100, np.nan)
    agg["FBPct"] = np.where(n > 0, agg["fb"] / n * 100, np.nan)
    agg["LDPct"] = np.where(n > 0, agg["ld"] / n * 100, np.nan)
    agg["PUPct"] = np.where(n > 0, agg["pu"] / n * 100, np.nan)
    agg["AirPct"] = np.where(n > 0, (agg["fb"] + agg["ld"] + agg["pu"]) / n * 100, np.nan)
    agg["BBE"] = agg["bbe"]

    # Directional stats (pull/center/oppo) — requires per-batter side, compute for batted balls
    if not batted.empty and "Direction" in batted.columns:
        batter_side = df.groupby(["Batter", "BatterTeam"])["BatterSide"].agg(
            lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else "Right"
        ).reset_index().rename(columns={"BatterSide": "_side"})
        dir_df = batted.dropna(subset=["Direction"]).merge(batter_side, on=["Batter", "BatterTeam"], how="left")
        dir_df["_side"] = dir_df["_side"].fillna("Right")
        dir_df["_pull"] = np.where(dir_df["_side"] == "Left", dir_df["Direction"] > 15, dir_df["Direction"] < -15)
        dir_df["_oppo"] = np.where(dir_df["_side"] == "Left", dir_df["Direction"] < -15, dir_df["Direction"] > 15)
        dir_agg = dir_df.groupby(["Batter", "BatterTeam"]).agg(
            _n_dir=("_pull", "size"), _pull=("_pull", "sum"), _oppo=("_oppo", "sum"),
        ).reset_index()
        dir_agg["PullPct"] = np.where(dir_agg["_n_dir"] > 0, dir_agg["_pull"] / dir_agg["_n_dir"] * 100, np.nan)
        dir_agg["OppoPct"] = np.where(dir_agg["_n_dir"] > 0, dir_agg["_oppo"] / dir_agg["_n_dir"] * 100, np.nan)
        dir_agg["StraightPct"] = np.where(dir_agg["_n_dir"] > 0, 100 - dir_agg["PullPct"] - dir_agg["OppoPct"], np.nan)
        agg = agg.merge(dir_agg[["Batter", "BatterTeam", "PullPct", "OppoPct", "StraightPct"]],
                        on=["Batter", "BatterTeam"], how="left")
    else:
        agg["PullPct"] = np.nan
        agg["OppoPct"] = np.nan
        agg["StraightPct"] = np.nan

    # Select output columns
    keep = ["Batter", "BatterTeam", "PA", "BBE", "AvgEV", "MaxEV", "HardHitPct",
            "Barrels", "BarrelPct", "BarrelPA", "SweetSpotPct", "AvgLA", "AvgDist",
            "WhiffPct", "KPct", "BBPct", "ChasePct", "ChaseContact",
            "ZoneSwingPct", "ZoneContactPct", "ZonePct", "SwingPct",
            "GBPct", "FBPct", "LDPct", "PUPct", "AirPct",
            "PullPct", "StraightPct", "OppoPct"]
    return agg[[c for c in keep if c in agg.columns]]


@st.cache_data(show_spinner=False)
def compute_pitcher_stats(data, season_filter=None):
    df = data.copy()
    if season_filter:
        df = df[df["Season"].isin(season_filter)]
    batter_zones = _build_batter_zones(df)
    _in_zone = in_zone_mask(df, batter_zones, batter_col="Batter")
    has_loc = df["PlateLocSide"].notna() & df["PlateLocHeight"].notna()
    out_zone = ~_in_zone & has_loc

    # Build unique PA identifier (robust to sampled data where PitchofPA==1 may be missing)
    _pa_id_cols = ["GameID", "Inning", "PAofInning", "Batter"]
    if all(c in df.columns for c in _pa_id_cols):
        df["_pa_id"] = (df["GameID"].astype(str) + "_" + df["Inning"].astype(str)
                        + "_" + df["PAofInning"].astype(str) + "_" + df["Batter"].astype(str))
    else:
        df["_pa_id"] = df["PitchofPA"].astype(str) + "_" + df.index.astype(str)

    # Pre-compute boolean columns
    df["_is_swing"] = df["PitchCall"].isin(SWING_CALLS)
    df["_is_whiff"] = df["PitchCall"] == "StrikeSwinging"
    df["_is_contact"] = df["PitchCall"].isin(CONTACT_CALLS)
    df["_is_inplay"] = df["PitchCall"] == "InPlay"
    df["_in_zone"] = _in_zone
    df["_out_zone"] = out_zone
    df["_has_loc"] = has_loc
    df["_iz_swing"] = _in_zone & df["_is_swing"]
    df["_oz_swing"] = out_zone & df["_is_swing"]
    df["_iz_contact"] = _in_zone & df["_is_contact"]
    df["_has_ev"] = df["_is_inplay"] & df["ExitSpeed"].notna()
    df["_hard_hit"] = df["_has_ev"] & (df["ExitSpeed"] >= 95)
    df["_is_fb"] = df["TaggedPitchType"].isin(["Fastball", "Sinker", "Cutter"])
    df["_fb_velo"] = np.where(df["_is_fb"] & df["RelSpeed"].notna(), df["RelSpeed"], np.nan)

    # Count unique PAs and deduplicated K/BB per pitcher
    pa_counts = df.groupby(["Pitcher", "PitcherTeam"])["_pa_id"].nunique().reset_index(name="PA")
    k_pas = df[df["KorBB"] == "Strikeout"].groupby(["Pitcher", "PitcherTeam"])["_pa_id"].nunique().reset_index(name="ks")
    bb_pas = df[df["KorBB"] == "Walk"].groupby(["Pitcher", "PitcherTeam"])["_pa_id"].nunique().reset_index(name="bbs")

    # Vectorized groupby
    grp = df.groupby(["Pitcher", "PitcherTeam"])
    agg = grp.agg(
        Pitches=("_is_swing", "size"),
        swings=("_is_swing", "sum"),
        whiffs=("_is_whiff", "sum"),
        n_loc=("_has_loc", "sum"),
        iz_count=("_in_zone", "sum"),
        oz_count=("_out_zone", "sum"),
        iz_swings=("_iz_swing", "sum"),
        oz_swings=("_oz_swing", "sum"),
        iz_contacts=("_iz_contact", "sum"),
        bbe=("_has_ev", "sum"),
        hard_hits=("_hard_hit", "sum"),
        AvgFBVelo=("_fb_velo", "mean"),
        MaxFBVelo=("_fb_velo", "max"),
        AvgSpin=("SpinRate", "mean"),
        Extension=("Extension", "mean"),
    ).reset_index()

    # Merge correct PA and K/BB counts
    agg = agg.merge(pa_counts, on=["Pitcher", "PitcherTeam"], how="left")
    agg = agg.merge(k_pas, on=["Pitcher", "PitcherTeam"], how="left")
    agg = agg.merge(bb_pas, on=["Pitcher", "PitcherTeam"], how="left")
    agg["PA"] = agg["PA"].fillna(0).astype(int)
    agg["ks"] = agg["ks"].fillna(0).astype(int)
    agg["bbs"] = agg["bbs"].fillna(0).astype(int)

    agg = agg[agg["Pitches"] >= 100].copy()

    # Batted ball sub-aggregations
    batted = df[df["_has_ev"]].copy()
    if not batted.empty:
        batted["_barrel"] = is_barrel_mask(batted)
        ba = batted.groupby(["Pitcher", "PitcherTeam"]).agg(
            AvgEVAgainst=("ExitSpeed", "mean"),
            n_barrels=("_barrel", "sum"),
        ).reset_index()
        agg = agg.merge(ba, on=["Pitcher", "PitcherTeam"], how="left")
    else:
        agg["AvgEVAgainst"] = np.nan
        agg["n_barrels"] = 0

    # GB count from in-play
    inplay = df[df["_is_inplay"]].copy()
    if not inplay.empty and "TaggedHitType" in inplay.columns:
        gb_counts = inplay[inplay["TaggedHitType"] == "GroundBall"].groupby(["Pitcher", "PitcherTeam"]).size().reset_index(name="gb")
        ip_counts = inplay.groupby(["Pitcher", "PitcherTeam"]).size().reset_index(name="n_ip")
        agg = agg.merge(gb_counts, on=["Pitcher", "PitcherTeam"], how="left")
        agg = agg.merge(ip_counts, on=["Pitcher", "PitcherTeam"], how="left")
        agg["gb"] = agg["gb"].fillna(0)
        agg["n_ip"] = agg["n_ip"].fillna(0)
    else:
        agg["gb"] = 0
        agg["n_ip"] = 0

    # Compute derived percentages
    pa = agg["PA"]
    sw = agg["swings"]
    oz = agg["oz_count"]
    iz_sw = agg["iz_swings"]
    n_bat = agg["bbe"]
    n_ip = agg["n_ip"]

    agg["WhiffPct"] = np.where(sw > 0, agg["whiffs"] / sw * 100, np.nan)
    agg["KPct"] = np.where(pa > 0, agg["ks"] / pa * 100, np.nan)
    agg["BBPct"] = np.where(pa > 0, agg["bbs"] / pa * 100, np.nan)
    agg["ZonePct"] = np.where(agg["n_loc"] > 0, agg["iz_count"] / agg["n_loc"] * 100, np.nan)
    agg["ChasePct"] = np.where(oz > 0, agg["oz_swings"] / oz * 100, np.nan)
    agg["ZoneContactPct"] = np.where(iz_sw > 0, agg["iz_contacts"] / iz_sw * 100, np.nan)
    agg["HardHitAgainst"] = np.where(n_bat > 0, agg["hard_hits"] / n_bat * 100, np.nan)
    agg["BarrelPctAgainst"] = np.where(n_bat > 0, agg["n_barrels"] / n_bat * 100, np.nan)
    agg["GBPct"] = np.where(n_ip > 0, agg["gb"] / n_ip * 100, np.nan)
    agg["SwingPct"] = np.where(agg["Pitches"] > 0, sw / agg["Pitches"] * 100, np.nan)

    keep = ["Pitcher", "PitcherTeam", "Pitches", "PA", "AvgFBVelo", "MaxFBVelo",
            "AvgSpin", "Extension", "WhiffPct", "KPct", "BBPct", "ZonePct",
            "ChasePct", "ZoneContactPct", "AvgEVAgainst", "HardHitAgainst",
            "BarrelPctAgainst", "GBPct", "SwingPct"]
    return agg[[c for c in keep if c in agg.columns]]


def get_percentile(value, series):
    if pd.isna(value) or series.dropna().empty:
        return np.nan
    return percentileofscore(series.dropna(), value, kind='rank')


def display_name(name, escape_html=True):
    if not name:
        return "Unknown"
    parts = name.split(", ")
    result = f"{parts[1]} {parts[0]}" if len(parts) == 2 else name
    return html_mod.escape(result) if escape_html else result


# ──────────────────────────────────────────────
# SAVANT-STYLE PERCENTILE BAR (color-gradient)
# ──────────────────────────────────────────────
def savant_color(pct, higher_is_better=True):
    """Statcast-style gradient: blue(poor) -> gray(avg) -> red(great)"""
    if pd.isna(pct):
        return "#aaa"
    p = pct if higher_is_better else (100 - pct)
    if p >= 95: return "#be0000"
    if p >= 90: return "#d22d49"
    if p >= 80: return "#e65730"
    if p >= 70: return "#ee7e1e"
    if p >= 60: return "#d4a017"
    if p >= 40: return "#9e9e9e"
    if p >= 30: return "#6a9bc3"
    if p >= 20: return "#3d7dab"
    if p >= 10: return "#1f5f8b"
    return "#14365d"


def _pctile_text_color(bg_color):
    """Return white or dark text depending on background brightness."""
    # Dark backgrounds need white text; light/mid backgrounds need dark text
    dark_bgs = {"#14365d", "#1f5f8b", "#3d7dab", "#be0000", "#d22d49", "#e65730"}
    return "#ffffff" if bg_color in dark_bgs else "#1a1a2e"


def render_savant_percentile_section(metrics_data, title=None, legend=None):
    """Render Baseball Savant style percentile ranking section.
    metrics_data: list of (label, value, percentile, fmt, higher_is_better)
    legend: optional tuple of (left_label, center_label, right_label) to override POOR/AVERAGE/GREAT
    """
    if title:
        st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

    # Legend
    l_left, l_center, l_right = legend or ("POOR", "AVERAGE", "GREAT")
    st.markdown(
        '<div style="display:flex;justify-content:space-between;margin-bottom:8px;padding:0 4px;">'
        f'<span style="font-size:10px;font-weight:700;color:#14365d !important;letter-spacing:0.5px;">{l_left}</span>'
        f'<span style="font-size:10px;font-weight:700;color:#9e9e9e !important;letter-spacing:0.5px;">{l_center}</span>'
        f'<span style="font-size:10px;font-weight:700;color:#be0000 !important;letter-spacing:0.5px;">{l_right}</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    for label, val, pct, fmt, hib in metrics_data:
        color = savant_color(pct, hib)
        txt_color = _pctile_text_color(color)
        # For "lower is better" metrics, invert position & displayed number
        effective_pct = pct if hib else (100 - pct) if not pd.isna(pct) else pct
        display_pct = int(round(effective_pct)) if not pd.isna(pct) else "-"
        display_val = f"{val:{fmt}}" if not pd.isna(val) else "-"
        bar_left = max(min(effective_pct, 100), 0) if not pd.isna(pct) else 50

        st.markdown(
            f'<div style="display:flex;align-items:center;margin:4px 0;height:30px;background:white;'
            f'border-radius:4px;padding:0 8px;">'
            # Label
            f'<div style="min-width:110px;font-size:11px;font-weight:600;color:#1a1a2e !important;'
            f'white-space:nowrap;text-transform:uppercase;letter-spacing:0.3px;">{label}</div>'
            # Percentile circle on gradient bar
            f'<div style="flex:1;position:relative;height:6px;'
            f'background:linear-gradient(to right, #14365d 0%, #3d7dab 25%, #9e9e9e 50%, #ee7e1e 75%, #be0000 100%);'
            f'border-radius:3px;margin:0 12px;">'
            f'<div style="position:absolute;left:{bar_left}%;top:50%;transform:translate(-50%,-50%);'
            f'width:28px;height:28px;border-radius:50%;background:{color};border:3px solid white;'
            f'box-shadow:0 1px 4px rgba(0,0,0,0.3);display:flex;align-items:center;justify-content:center;'
            f'font-size:10px;font-weight:800;color:{txt_color} !important;'
            f'text-shadow:0 0 2px rgba(0,0,0,0.3);">{display_pct}</div>'
            f'</div>'
            # Value
            f'<div style="min-width:50px;text-align:right;font-size:12px;font-weight:700;color:#1a1a2e !important;">'
            f'{display_val}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────
# CHART HELPERS
# ──────────────────────────────────────────────
def strike_zone_rect():
    return dict(type="rect", x0=-ZONE_SIDE, x1=ZONE_SIDE, y0=ZONE_HEIGHT_BOT, y1=ZONE_HEIGHT_TOP,
                line=dict(color="#333", width=2), fillcolor="rgba(0,0,0,0)")


def add_strike_zone(fig, label=True):
    fig.add_shape(strike_zone_rect())
    if label:
        fig.add_annotation(
            x=0, y=ZONE_HEIGHT_BOT - 0.18, text="Catcher's View",
            showarrow=False, font=dict(size=9, color="#999"),
            xanchor="center", yanchor="top",
        )
    return fig


def _add_grid_zone_outline(fig, color="#333", width=3):
    """Add a rectangular strike zone outline to a 5x5 categorical grid heatmap.

    The inner 3x3 (indices 1-3) is the strike zone.  Uses a simple rectangle
    matching the style of the continuous-axis heatmaps (contact rate, damage).
    """
    fig.add_shape(
        type="rect",
        x0=0.5, y0=0.5, x1=3.5, y1=3.5,
        line=dict(color=color, width=width),
        fillcolor="rgba(0,0,0,0)",
    )
    return fig


CHART_LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(size=11, color="#1a1a2e", family="Inter, Arial, sans-serif"),
    margin=dict(l=45, r=10, t=30, b=40),
)


def make_spray_chart(in_play_df, height=360):
    spray = in_play_df.dropna(subset=["Direction", "Distance"]).copy()
    if spray.empty:
        return None
    angle_rad = np.radians(spray["Direction"])
    spray["x"] = spray["Distance"] * np.sin(angle_rad)
    spray["y"] = spray["Distance"] * np.cos(angle_rad)

    fig = go.Figure()

    # Grass fill
    theta_grass = np.linspace(-np.pi / 4, np.pi / 4, 80)
    grass_r = 400
    grass_x = [0] + list(grass_r * np.sin(theta_grass)) + [0]
    grass_y = [0] + list(grass_r * np.cos(theta_grass)) + [0]
    fig.add_trace(go.Scatter(x=grass_x, y=grass_y, mode="lines",
                             fill="toself", fillcolor="rgba(76,160,60,0.06)",
                             line=dict(color="rgba(76,160,60,0.15)", width=1), showlegend=False,
                             hoverinfo="skip"))

    # Infield
    diamond_x = [0, 63.6, 0, -63.6, 0]
    diamond_y = [0, 63.6, 127.3, 63.6, 0]
    fig.add_trace(go.Scatter(x=diamond_x, y=diamond_y, mode="lines",
                             line=dict(color="rgba(160,120,60,0.25)", width=1), showlegend=False,
                             fill="toself", fillcolor="rgba(160,120,60,0.06)", hoverinfo="skip"))

    # Foul lines
    fl = 350
    fig.add_trace(go.Scatter(x=[0, -fl * np.sin(np.pi / 4)], y=[0, fl * np.cos(np.pi / 4)],
                             mode="lines", line=dict(color="rgba(0,0,0,0.12)", width=1),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0, fl * np.sin(np.pi / 4)], y=[0, fl * np.cos(np.pi / 4)],
                             mode="lines", line=dict(color="rgba(0,0,0,0.12)", width=1),
                             showlegend=False, hoverinfo="skip"))

    # Hit type colors
    ht_colors = {"GroundBall": "#d62728", "LineDrive": "#2ca02c", "FlyBall": "#1f77b4", "Popup": "#ff7f0e"}
    ht_names = {"GroundBall": "GB", "LineDrive": "LD", "FlyBall": "FB", "Popup": "PU"}

    for ht in ["GroundBall", "LineDrive", "FlyBall", "Popup"]:
        sub = spray[spray["TaggedHitType"] == ht]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["x"], y=sub["y"], mode="markers",
            marker=dict(size=7, color=ht_colors.get(ht, "#777"), opacity=0.8,
                        line=dict(width=0.5, color="white")),
            name=ht_names.get(ht, ht),
            hovertemplate="EV: %{customdata[0]:.1f}<br>Dist: %{customdata[1]:.0f}ft<extra></extra>",
            customdata=sub[["ExitSpeed", "Distance"]].fillna(0).values,
        ))

    other = spray[~spray["TaggedHitType"].isin(["GroundBall", "LineDrive", "FlyBall", "Popup"])]
    if not other.empty:
        fig.add_trace(go.Scatter(x=other["x"], y=other["y"], mode="markers",
                                 marker=dict(size=5, color="#bbb", opacity=0.5), name="Other"))

    fig.update_layout(
        xaxis=dict(range=[-300, 300], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-15, 400], showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", fixedrange=True),
        height=height, margin=dict(l=0, r=0, t=5, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="#1a1a2e", family="Inter, Arial, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5,
                    font=dict(size=10, color="#1a1a2e"), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def player_header(name, jersey, pos, detail_line, stat_line):
    st.markdown(
        f'<div class="player-header-dark" style="background:linear-gradient(135deg,#000000,#1a1a1a,#000000);'
        f'padding:20px 28px;border-radius:10px;margin-bottom:16px;border:1px solid rgba(255,255,255,0.1);">'
        f'<div style="display:flex;align-items:center;gap:20px;">'
        f'<div class="ph-jersey" style="font-size:48px;font-weight:900;line-height:1;'
        f'font-family:Inter,sans-serif;text-shadow:0 2px 4px rgba(0,0,0,0.3);">#{jersey}</div>'
        f'<div>'
        f'<div class="ph-name" style="font-size:30px;font-weight:800;margin:0;'
        f'font-family:Inter,sans-serif;text-shadow:0 1px 3px rgba(0,0,0,0.3);">{display_name(name)}</div>'
        f'<div class="ph-detail" style="font-size:14px;margin-top:3px;font-family:Inter,sans-serif;">'
        f'{detail_line}</div>'
        f'<div class="ph-stat" style="font-size:13px;margin-top:2px;font-family:Inter,sans-serif;">'
        f'{stat_line}</div>'
        f'</div></div></div>',
        unsafe_allow_html=True,
    )


def section_header(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# PAGE: HITTER CARD (Statcast Style)
# ──────────────────────────────────────────────
def _hitter_card_content(data, batter, season_filter, bdf, batted, pr, all_batter_stats):
    """Render a single-page Hitter Card — simple, actionable summary."""
    all_stats = all_batter_stats
    in_play = bdf[bdf["PitchCall"] == "InPlay"]
    swings = bdf[bdf["PitchCall"].isin(SWING_CALLS)]
    whiffs = bdf[bdf["PitchCall"] == "StrikeSwinging"]
    _bs = safe_mode(bdf["BatterSide"], "Right") if "BatterSide" in bdf.columns else "Right"

    # ── ROW 1: Percentile Rankings + Spray Chart ──
    pc_col1, pc_col2 = st.columns([1, 1], gap="medium")
    with pc_col1:
        batting_metrics = [
            ("Avg EV", pr["AvgEV"], get_percentile(pr["AvgEV"], all_stats["AvgEV"]), ".1f", True),
            ("Max EV", pr["MaxEV"], get_percentile(pr["MaxEV"], all_stats["MaxEV"]), ".1f", True),
            ("Barrel %", pr["BarrelPct"], get_percentile(pr["BarrelPct"], all_stats["BarrelPct"]), ".1f", True),
            ("Hard Hit %", pr["HardHitPct"], get_percentile(pr["HardHitPct"], all_stats["HardHitPct"]), ".1f", True),
            ("Sweet Spot %", pr["SweetSpotPct"], get_percentile(pr["SweetSpotPct"], all_stats["SweetSpotPct"]), ".1f", True),
            ("K %", pr["KPct"], get_percentile(pr["KPct"], all_stats["KPct"]), ".1f", False),
            ("BB %", pr["BBPct"], get_percentile(pr["BBPct"], all_stats["BBPct"]), ".1f", True),
            ("Whiff %", pr["WhiffPct"], get_percentile(pr["WhiffPct"], all_stats["WhiffPct"]), ".1f", False),
            ("Chase %", pr["ChasePct"], get_percentile(pr["ChasePct"], all_stats["ChasePct"]), ".1f", False),
            ("Z-Contact %", pr["ZoneContactPct"], get_percentile(pr["ZoneContactPct"], all_stats["ZoneContactPct"]), ".1f", True),
        ]
        render_savant_percentile_section(batting_metrics, "Percentile Rankings")
        st.caption(f"vs. {len(all_stats)} batters in database (min 50 PA)")

    with pc_col2:
        section_header("Spray Chart")
        fig_spray = make_spray_chart(in_play, height=420)
        if fig_spray:
            st.plotly_chart(fig_spray, use_container_width=True, key="hc_spray")
        else:
            st.info("No batted ball data.")

        # Batted ball profile cards underneath (like pitcher usage/velo cards)
        bb_items = [
            ("Pull", f"{pr['PullPct']:.0f}%" if not pd.isna(pr.get('PullPct')) else "-", "#e63946"),
            ("Center", f"{pr['StraightPct']:.0f}%" if not pd.isna(pr.get('StraightPct')) else "-", "#457b9d"),
            ("Oppo", f"{pr['OppoPct']:.0f}%" if not pd.isna(pr.get('OppoPct')) else "-", "#2a9d8f"),
            ("GB", f"{pr['GBPct']:.0f}%" if not pd.isna(pr.get('GBPct')) else "-", "#d62728"),
            ("FB/LD", f"{(pr.get('FBPct', 0) or 0) + (pr.get('LDPct', 0) or 0):.0f}%" if not pd.isna(pr.get('FBPct')) else "-", "#1f77b4"),
        ]
        bb_cols = st.columns(len(bb_items))
        for idx, (name, val, color) in enumerate(bb_items):
            with bb_cols[idx]:
                st.markdown(
                    f'<div style="text-align:center;padding:6px 4px;border-radius:6px;'
                    f'border-top:3px solid {color};background:#f9f9f9;">'
                    f'<div style="font-weight:bold;font-size:13px;color:{color};">{name}</div>'
                    f'<div style="font-size:12px;color:#555;">{val}</div>'
                    f'</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── ROW 2: Damage Heatmap + Whiff Density + Swing Probability ──
    section_header("Where They Whiff & Where They Crush")
    loc_data = bdf.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    whiff_loc = loc_data[loc_data["PitchCall"] == "StrikeSwinging"]
    contacts_loc = loc_data[loc_data["PitchCall"].isin(CONTACT_CALLS)]
    batted_loc = loc_data[(loc_data["PitchCall"] == "InPlay")].dropna(subset=["ExitSpeed"])

    loc_cols = st.columns(3)

    # 1) Damage Heatmap (Avg EV) — contour with barrel stars
    with loc_cols[0]:
        section_header("Damage Heatmap (Avg EV)")
        if len(batted_loc) >= 5:
            fig_dmg = go.Figure()
            fig_dmg.add_trace(go.Histogram2dContour(
                x=batted_loc["PlateLocSide"], y=batted_loc["PlateLocHeight"],
                z=batted_loc["ExitSpeed"], histfunc="avg",
                colorscale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                contours=dict(showlines=False), ncontours=12, showscale=True,
                colorbar=dict(title="Avg EV", len=0.8),
            ))
            barrel_loc = batted_loc[is_barrel_mask(batted_loc)]
            if not barrel_loc.empty:
                fig_dmg.add_trace(go.Scatter(
                    x=barrel_loc["PlateLocSide"], y=barrel_loc["PlateLocHeight"],
                    mode="markers", marker=dict(size=12, color="#d22d49", symbol="star",
                                                 line=dict(width=1, color="white")),
                    name="Barrels", hovertemplate="EV: %{customdata[0]:.1f}<extra></extra>",
                    customdata=barrel_loc[["ExitSpeed"]].values))
            add_strike_zone(fig_dmg)
            fig_dmg.update_layout(**CHART_LAYOUT, height=350,
                                   xaxis=dict(range=[-1.8, 1.8], title="Horizontal", scaleanchor="y"),
                                   yaxis=dict(range=[0.5, 4.5], title="Vertical"),
                                   legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font=dict(size=10)))
            st.plotly_chart(fig_dmg, use_container_width=True, key="hc_damage")
        else:
            st.caption("Not enough batted ball data")
        st.caption(f"{len(batted_loc)} batted balls")

    # 2) Whiff Zone Map — 5x5 grid heatmap with whiff% per zone
    with loc_cols[1]:
        section_header("Whiff Zone Map")
        grid_whiff, annot_whiff, h_lbl, v_lbl = _create_zone_grid_data(bdf, metric="whiff_rate", batter_side=_bs)
        if not np.isnan(grid_whiff).all():
            fig_wz = go.Figure(data=go.Heatmap(
                z=grid_whiff, text=annot_whiff, texttemplate="%{text}",
                x=h_lbl, y=v_lbl,
                colorscale=[[0, "#2ca02c"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                zmin=0, zmax=60, showscale=True,
                colorbar=dict(title="Whiff%", len=0.8),
                textfont=dict(size=13, color="white"),
            ))
            _add_grid_zone_outline(fig_wz)
            fig_wz.update_layout(**CHART_LAYOUT, height=350)
            st.plotly_chart(fig_wz, use_container_width=True, key="hc_whiff_zone")
        else:
            st.caption("Not enough swing data for whiff zones")

    # 3) Swing Probability Contour — P(Swing) by location
    with loc_cols[2]:
        section_header("Swing Probability")
        all_with_loc = loc_data.copy()
        all_with_loc["is_swing"] = all_with_loc["PitchCall"].isin(SWING_CALLS).astype(int)
        if len(all_with_loc) >= 20:
            fig_prob = go.Figure()
            fig_prob.add_trace(go.Histogram2dContour(
                x=all_with_loc["PlateLocSide"], y=all_with_loc["PlateLocHeight"],
                z=all_with_loc["is_swing"],
                histfunc="avg",
                colorscale=[[0, "#2ca02c"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                contours=dict(showlabels=True, labelfont=dict(size=10)),
                nbinsx=12, nbinsy=12,
                showscale=True,
                colorbar=dict(title="P(Swing)", len=0.8),
            ))
            add_strike_zone(fig_prob)
            fig_prob.update_layout(**CHART_LAYOUT, height=350,
                                    xaxis=dict(range=[-1.8, 1.8], title="Horizontal", scaleanchor="y"),
                                    yaxis=dict(range=[0.5, 4.5], title="Vertical"))
            st.plotly_chart(fig_prob, use_container_width=True, key="hc_swing_prob")
        else:
            st.caption("Not enough pitch data")
        st.caption(f"{len(all_with_loc)} pitches")

    st.markdown("---")

    # ── ROW 3: Best Count Performance + Pitch Type Performance ──
    col_counts, col_pitch = st.columns([1, 1], gap="medium")

    with col_counts:
        section_header("Count Performance")
        if "Balls" in bdf.columns and "Strikes" in bdf.columns:
            bdf_counts = bdf.dropna(subset=["Balls", "Strikes"]).copy()
            bdf_counts["Count"] = bdf_counts["Balls"].astype(int).astype(str) + "-" + bdf_counts["Strikes"].astype(int).astype(str)
            all_data_counts = data.dropna(subset=["Balls", "Strikes"]).copy()
            all_data_counts["Count"] = all_data_counts["Balls"].astype(int).astype(str) + "-" + all_data_counts["Strikes"].astype(int).astype(str)

            def _cs(sub):
                sw = sub[sub["PitchCall"].isin(SWING_CALLS)]
                wh = sub[sub["PitchCall"] == "StrikeSwinging"]
                ip = sub[sub["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
                return {
                    "n": len(sub), "swings": len(sw),
                    "whiff_pct": len(wh) / max(len(sw), 1) * 100 if len(sw) > 0 else 0,
                    "ev": ip["ExitSpeed"].mean() if len(ip) > 0 else np.nan,
                }

            db_avgs = {cnt: _cs(grp) for cnt, grp in all_data_counts.groupby("Count")}
            player_cs = {cnt: _cs(grp) for cnt, grp in bdf_counts.groupby("Count")}

            # Avg EV heatmap by count
            balls_range = [0, 1, 2, 3]
            strikes_range = [0, 1, 2]
            z_vals, hover_text, annotations = [], [], []
            for s in strikes_range:
                row_z, row_hover = [], []
                for b in balls_range:
                    cnt = f"{b}-{s}"
                    ps = player_cs.get(cnt, {})
                    db = db_avgs.get(cnt, {})
                    p_ev = ps.get("ev", np.nan)
                    db_ev = db.get("ev", np.nan)
                    n = ps.get("n", 0)
                    p_whiff = ps.get("whiff_pct", np.nan)
                    if n >= 3 and not pd.isna(p_ev):
                        diff = p_ev - db_ev if not pd.isna(db_ev) else 0
                        row_z.append(diff)
                        row_hover.append(
                            f"<b>{cnt}</b> ({n} pitches)<br>"
                            f"Avg EV: <b>{p_ev:.1f} mph</b><br>"
                            f"DB Avg: {db_ev:.1f} mph<br>"
                            f"Whiff%: {p_whiff:.1f}%"
                        )
                        annotations.append(dict(
                            x=balls_range.index(b), y=strikes_range.index(s),
                            text=f"<b>{p_ev:.0f}</b><br><span style='font-size:9px'>W: {p_whiff:.0f}%</span>",
                            showarrow=False, font=dict(size=13, color="white", family="Inter"),
                        ))
                    else:
                        row_z.append(np.nan)
                        row_hover.append(f"{cnt}<br>Not enough data")
                        annotations.append(dict(
                            x=balls_range.index(b), y=strikes_range.index(s),
                            text="—", showarrow=False, font=dict(size=13, color="#aaa"),
                        ))
                z_vals.append(row_z)
                hover_text.append(row_hover)

            fig = go.Figure(data=go.Heatmap(
                z=z_vals,
                x=[f"{b} Balls" for b in balls_range],
                y=[f"{s} Strikes" for s in strikes_range],
                colorscale=[
                    [0.0, "#cc0000"], [0.35, "#e8a0a0"], [0.5, "#f0f0f0"],
                    [0.65, "#a0d4a0"], [1.0, "#1a7a1a"],
                ],
                zmid=0, showscale=False,
                hovertext=hover_text, hoverinfo="text",
                xgap=3, ygap=3,
            ))
            fig.update_layout(
                annotations=annotations,
                xaxis=dict(side="top", tickfont=dict(size=12, color="#1a1a2e")),
                yaxis=dict(tickfont=dict(size=12, color="#1a1a2e"), autorange="reversed"),
                height=250, margin=dict(l=80, r=20, t=40, b=20),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Inter, Arial, sans-serif"),
            )
            st.plotly_chart(fig, use_container_width=True, key="hc_count_grid")
            st.caption("EV (bold) + Whiff% per count. Green = EV above DB avg, Red = below.")
        else:
            st.caption("Count data not available.")

    with col_pitch:
        section_header("Performance vs Pitch Types")
        pitch_groups = {
            "Fastball": ["Fastball", "Sinker", "Cutter"],
            "Breaking": ["Slider", "Curveball", "Sweeper", "Knuckle Curve"],
            "Offspeed": ["Changeup", "Splitter"],
        }
        pt_rows = []
        for group_name, types in pitch_groups.items():
            sub = bdf[bdf["TaggedPitchType"].isin(types)]
            if len(sub) < 3:
                continue
            sub_swings = sub[sub["PitchCall"].isin(SWING_CALLS)]
            sub_whiffs = sub[sub["PitchCall"] == "StrikeSwinging"]
            sub_ip = sub[sub["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            whiff_pct = len(sub_whiffs) / max(len(sub_swings), 1) * 100
            avg_ev = sub_ip["ExitSpeed"].mean() if len(sub_ip) > 0 else np.nan
            hard_pct = len(sub_ip[sub_ip["ExitSpeed"] >= 95]) / max(len(sub_ip), 1) * 100 if len(sub_ip) > 0 else np.nan
            pt_rows.append({
                "Pitch Group": group_name,
                "#": len(sub),
                "Seen %": round(len(sub) / len(bdf) * 100, 1),
                "Whiff%": round(whiff_pct, 1),
                "Avg EV": round(avg_ev, 1) if not pd.isna(avg_ev) else None,
                "Hard%": round(hard_pct, 1) if not pd.isna(hard_pct) else None,
            })
        if pt_rows:
            st.dataframe(pd.DataFrame(pt_rows), use_container_width=True, hide_index=True)

        # Detailed per-pitch breakdown
        bdf_filt = filter_minor_pitches(bdf)
        detail_rows = []
        for pt in sorted(bdf_filt["TaggedPitchType"].dropna().unique()):
            sub = bdf_filt[bdf_filt["TaggedPitchType"] == pt]
            if len(sub) < 3:
                continue
            sub_sw = sub[sub["PitchCall"].isin(SWING_CALLS)]
            sub_wh = sub[sub["PitchCall"] == "StrikeSwinging"]
            sub_ip = sub[sub["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            detail_rows.append({
                "Pitch": pt,
                "#": len(sub),
                "%": round(len(sub) / len(bdf) * 100, 1),
                "Whiff%": round(len(sub_wh) / max(len(sub_sw), 1) * 100, 1),
                "EV": round(sub_ip["ExitSpeed"].mean(), 1) if len(sub_ip) > 0 else None,
                "LA": round(sub_ip["Angle"].mean(), 1) if len(sub_ip) > 0 and sub_ip["Angle"].notna().any() else None,
            })
        if detail_rows:
            with st.expander("Detailed Pitch Type Breakdown"):
                st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── ROW 4: Swing Path Summary + EV vs LA Scatter ──
    col_swing, col_evla = st.columns([1, 1], gap="medium")

    with col_swing:
        section_header("Swing Path Profile")
        inplay_full = in_play.dropna(subset=["Angle", "ExitSpeed", "PlateLocHeight"]).copy()
        if len(inplay_full) >= 10:
            from scipy import stats as sp_stats
            ev_75 = inplay_full["ExitSpeed"].quantile(0.75)
            hard_hit = inplay_full[inplay_full["ExitSpeed"] >= ev_75].copy()
            attack_angle = hard_hit["Angle"].median()
            avg_la_all = inplay_full["Angle"].median()

            # Vertical path adjustment
            if len(hard_hit) >= 3:
                v_slope, v_int, _, _, _ = sp_stats.linregress(
                    hard_hit["PlateLocHeight"].values, hard_hit["Angle"].values)
                mid_zone_aa = v_int + v_slope * 2.5
            else:
                v_slope = 0
                mid_zone_aa = attack_angle

            # Bat speed proxy — empirical model calibrated to sensor data
            # EV ≈ 0.2 * pitch_speed + 1.2 * bat_speed
            bat_speed_avg = np.nan
            if "RelSpeed" in hard_hit.columns and hard_hit["RelSpeed"].notna().any():
                hard_hit["BatSpeedProxy"] = (hard_hit["ExitSpeed"] - 0.2 * hard_hit["RelSpeed"]) / 1.2
                bat_speed_avg = hard_hit["BatSpeedProxy"].mean()

            # Contact depth
            depth_label = None
            if "EffectiveVelo" in hard_hit.columns and hard_hit["EffectiveVelo"].notna().any():
                contact_depth = (hard_hit["EffectiveVelo"] - hard_hit["RelSpeed"]).mean()
                if contact_depth > 0:
                    depth_label = "Out Front"
                elif contact_depth > -1.5:
                    depth_label = "Neutral"
                else:
                    depth_label = "Deep Contact"

            # Classification
            if mid_zone_aa > 14:
                swing_type = "Uppercut / Lift"
            elif mid_zone_aa > 6:
                swing_type = "Positive / Line-Drive"
            elif mid_zone_aa > -2:
                swing_type = "Level"
            else:
                swing_type = "Negative / Chop"

            # Display as metric cards
            cards = [
                ("Attack Angle", f"{mid_zone_aa:.1f}°", swing_type),
                ("Path Adjust", f"{v_slope:.1f}°/ft", "Low = flat, High = adaptable"),
                ("Median LA", f"{avg_la_all:.1f}°", f"Hard-hit: {attack_angle:.1f}°"),
            ]
            if not pd.isna(bat_speed_avg):
                cards.append(("Bat Speed (est)", f"{bat_speed_avg:.1f} mph", "Top-25% EV proxy"))
            if depth_label:
                cards.append(("Contact Depth", depth_label, "Where bat meets ball"))

            for label, val, desc in cards:
                st.markdown(
                    f'<div style="display:flex;align-items:center;margin:4px 0;padding:8px 12px;'
                    f'background:#f9f9f9;border-radius:6px;border-left:3px solid #e63946;">'
                    f'<div style="flex:1;">'
                    f'<div style="font-size:11px;color:#888;text-transform:uppercase;">{label}</div>'
                    f'<div style="font-size:18px;font-weight:700;color:#1a1a2e;">{val}</div>'
                    f'</div>'
                    f'<div style="font-size:11px;color:#888;text-align:right;">{desc}</div>'
                    f'</div>', unsafe_allow_html=True)
        else:
            st.info("Not enough batted ball data for swing path analysis (need 10+).")

    with col_evla:
        section_header("Exit Velo vs Launch Angle")
        ev_la = in_play.dropna(subset=["ExitSpeed", "Angle"])
        if not ev_la.empty:
            fig = px.scatter(
                ev_la, x="Angle", y="ExitSpeed", color="TaggedHitType",
                color_discrete_map={"GroundBall": "#d62728", "LineDrive": "#2ca02c",
                                    "FlyBall": "#1f77b4", "Popup": "#ff7f0e"},
                opacity=0.7, labels={"Angle": "Launch Angle", "ExitSpeed": "Exit Velo"},
            )
            _bz_ev = np.linspace(98, max(ev_la["ExitSpeed"].max() + 2, 105), 40)
            _bz_la_lo = np.clip(26 - 2 * (_bz_ev - 98), 8, 26)
            _bz_la_hi = np.clip(30 + 3 * (_bz_ev - 98), 30, 50)
            fig.add_trace(go.Scatter(
                x=np.concatenate([_bz_la_lo, _bz_la_hi[::-1]]),
                y=np.concatenate([_bz_ev, _bz_ev[::-1]]),
                fill="toself", fillcolor="rgba(230,57,70,0.06)",
                line=dict(color="#e63946", width=1.5, dash="dash"),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_annotation(x=20, y=_bz_ev[-1], text="BARREL ZONE",
                               showarrow=False, font=dict(size=9, color="#e63946"))
            fig.update_layout(height=350, showlegend=True,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                          font=dict(size=9, color="#1a1a2e")),
                              **CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True, key="hc_ev_la")
        else:
            st.info("No exit velo / launch angle data.")

    # ── ROW 5: Platoon Splits ──
    st.markdown("---")
    section_header("Platoon Splits")
    if "PitcherThrows" in bdf.columns:
        split_metrics = []
        for side, label in [("Right", "vs RHP"), ("Left", "vs LHP")]:
            side_df = bdf[bdf["PitcherThrows"] == side]
            if len(side_df) < 5:
                continue
            s_sw = side_df[side_df["PitchCall"].isin(SWING_CALLS)]
            s_wh = side_df[side_df["PitchCall"] == "StrikeSwinging"]
            s_ip = side_df[side_df["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            s_whiff = len(s_wh) / max(len(s_sw), 1) * 100
            s_ev = s_ip["ExitSpeed"].mean() if len(s_ip) > 0 else np.nan
            s_hard = len(s_ip[s_ip["ExitSpeed"] >= 95]) / max(len(s_ip), 1) * 100 if len(s_ip) > 0 else np.nan

            # Percentiles vs all batters in same split (via DuckDB)
            adf = query_population(f"""
                SELECT
                    Batter,
                    CASE WHEN SUM(CASE WHEN PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END) > 0
                        THEN SUM(CASE WHEN PitchCall='StrikeSwinging' THEN 1 ELSE 0 END) * 100.0
                             / SUM(CASE WHEN PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END)
                        ELSE NULL END AS whiff,
                    AVG(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN ExitSpeed END) AS ev
                FROM trackman
                WHERE PitcherThrows = '{side}'
                GROUP BY Batter
                HAVING COUNT(*) >= 10
            """)
            w_pct = get_percentile(s_whiff, adf["whiff"]) if not adf.empty else 50
            e_pct = get_percentile(s_ev, adf["ev"]) if not adf.empty and not pd.isna(s_ev) else np.nan

            split_metrics.append((f"{label} Whiff%", s_whiff, w_pct, ".1f", False))
            if not pd.isna(e_pct):
                split_metrics.append((f"{label} Avg EV", s_ev, e_pct, ".1f", True))

        if split_metrics:
            render_savant_percentile_section(split_metrics, None)
            st.caption("Percentile vs. all batters in DB (min 10 pitches in that split)")
    else:
        st.caption("No pitcher-throws data available for splits.")


def _hitting_overview(data, batter, season_filter, bdf, batted, pr, all_batter_stats):
    """Content from the original Hitter Card, rendered inside the Overview tab."""
    in_play = bdf[bdf["PitchCall"] == "InPlay"]
    all_stats = all_batter_stats  # alias for brevity in this section

    # ── ROW 1: Percentile Rankings + Spray Chart + Batted Ball Stats ──
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        # ── PERCENTILE RANKINGS (Savant-style) ──
        batting_metrics = [
            ("Avg EV", pr["AvgEV"], get_percentile(pr["AvgEV"], all_stats["AvgEV"]), ".1f", True),
            ("Max EV", pr["MaxEV"], get_percentile(pr["MaxEV"], all_stats["MaxEV"]), ".1f", True),
            ("Barrel %", pr["BarrelPct"], get_percentile(pr["BarrelPct"], all_stats["BarrelPct"]), ".1f", True),
            ("Hard Hit %", pr["HardHitPct"], get_percentile(pr["HardHitPct"], all_stats["HardHitPct"]), ".1f", True),
            ("Sweet Spot %", pr["SweetSpotPct"], get_percentile(pr["SweetSpotPct"], all_stats["SweetSpotPct"]), ".1f", True),
            ("Avg LA", pr["AvgLA"], get_percentile(pr["AvgLA"], all_stats["AvgLA"]), ".1f", True),
            ("K %", pr["KPct"], get_percentile(pr["KPct"], all_stats["KPct"]), ".1f", False),
            ("BB %", pr["BBPct"], get_percentile(pr["BBPct"], all_stats["BBPct"]), ".1f", True),
            ("Whiff %", pr["WhiffPct"], get_percentile(pr["WhiffPct"], all_stats["WhiffPct"]), ".1f", False),
            ("Chase %", pr["ChasePct"], get_percentile(pr["ChasePct"], all_stats["ChasePct"]), ".1f", False),
        ]
        render_savant_percentile_section(batting_metrics, "Percentile Rankings")
        st.caption(f"vs. {len(all_stats)} batters in database (min 50 PA)")

    with col2:
        section_header("Hits Spray Chart")
        fig = make_spray_chart(in_play, height=420)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="hitter_spray")
        else:
            st.info("No batted ball data.")

    # ── ROW 2: Batting Stats Table + Quality of Contact ──
    st.markdown("---")
    col3, col4 = st.columns([1, 1], gap="medium")

    with col3:
        section_header("Statcast Batting Statistics")
        stats_df = pd.DataFrame([{
            "PA": int(pr["PA"]),
            "BBE": int(pr["BBE"]),
            "Barrels": int(pr["Barrels"]),
            "Barrel%": round(pr["BarrelPct"], 1) if not pd.isna(pr["BarrelPct"]) else None,
            "Brl/PA": round(pr["BarrelPA"], 1) if not pd.isna(pr["BarrelPA"]) else None,
            "Avg EV": round(pr["AvgEV"], 1) if not pd.isna(pr["AvgEV"]) else None,
            "Max EV": round(pr["MaxEV"], 1) if not pd.isna(pr["MaxEV"]) else None,
            "Avg LA": round(pr["AvgLA"], 1) if not pd.isna(pr["AvgLA"]) else None,
            "Sweet%": round(pr["SweetSpotPct"], 1) if not pd.isna(pr["SweetSpotPct"]) else None,
            "Hard%": round(pr["HardHitPct"], 1) if not pd.isna(pr["HardHitPct"]) else None,
            "K%": round(pr["KPct"], 1),
            "BB%": round(pr["BBPct"], 1),
        }])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # Batted Ball Profile
        section_header("Batted Ball Profile")
        bb_df = pd.DataFrame([{
            "GB%": round(pr["GBPct"], 1) if not pd.isna(pr["GBPct"]) else None,
            "Air%": round(pr["AirPct"], 1) if not pd.isna(pr["AirPct"]) else None,
            "FB%": round(pr["FBPct"], 1) if not pd.isna(pr["FBPct"]) else None,
            "LD%": round(pr["LDPct"], 1) if not pd.isna(pr["LDPct"]) else None,
            "PU%": round(pr["PUPct"], 1) if not pd.isna(pr["PUPct"]) else None,
            "Pull%": round(pr["PullPct"], 1) if not pd.isna(pr["PullPct"]) else None,
            "Cent%": round(pr["StraightPct"], 1) if not pd.isna(pr["StraightPct"]) else None,
            "Oppo%": round(pr["OppoPct"], 1) if not pd.isna(pr["OppoPct"]) else None,
        }])
        st.dataframe(bb_df, use_container_width=True, hide_index=True)

    with col4:
        section_header("Plate Discipline")
        disc_df = pd.DataFrame([{
            "Pitches": len(bdf),
            "Zone%": round(pr["ZonePct"], 1) if not pd.isna(pr["ZonePct"]) else None,
            "Z-Swing%": round(pr["ZoneSwingPct"], 1) if not pd.isna(pr["ZoneSwingPct"]) else None,
            "Z-Contact%": round(pr["ZoneContactPct"], 1) if not pd.isna(pr["ZoneContactPct"]) else None,
            "Chase%": round(pr["ChasePct"], 1),
            "Chase Ct%": round(pr["ChaseContact"], 1) if not pd.isna(pr["ChaseContact"]) else None,
            "Swing%": round(pr["SwingPct"], 1),
            "Whiff%": round(pr["WhiffPct"], 1),
            "K%": round(pr["KPct"], 1),
            "BB%": round(pr["BBPct"], 1),
        }])
        st.dataframe(disc_df, use_container_width=True, hide_index=True)

        # Quality of Contact
        section_header("Quality of Contact")
        if len(batted) > 0:
            weak = len(batted[batted["ExitSpeed"] < 70]) / len(batted) * 100
            topped = len(batted[batted["Angle"] < -10]) / len(batted) * 100 if batted["Angle"].notna().any() else 0
            under = len(batted[batted["Angle"] > 40]) / len(batted) * 100 if batted["Angle"].notna().any() else 0
            flare = len(batted[(batted["ExitSpeed"].between(70, 88)) & (batted["Angle"].between(8, 32))]) / len(batted) * 100 if batted["Angle"].notna().any() else 0
            solid = len(batted[(batted["ExitSpeed"].between(89, 97)) & (batted["Angle"].between(8, 32))]) / len(batted) * 100 if batted["Angle"].notna().any() else 0
            barrel = pr["BarrelPct"] if not pd.isna(pr["BarrelPct"]) else 0
            qoc_df = pd.DataFrame([{
                "Weak%": round(weak, 1),
                "Topped%": round(topped, 1),
                "Under%": round(under, 1),
                "Flare%": round(flare, 1),
                "Solid%": round(solid, 1),
                "Barrel%": round(barrel, 1),
                "Brl/PA": round(pr["BarrelPA"], 1) if not pd.isna(pr["BarrelPA"]) else None,
            }])
            st.dataframe(qoc_df, use_container_width=True, hide_index=True)

    # ── ROW 3: Rolling EV + EV vs LA + Swing Heatmap ──
    st.markdown("---")
    col5, col6, col7 = st.columns([1, 1, 1], gap="medium")

    with col5:
        section_header("Rolling Exit Velocity")
        batted_sorted = in_play.dropna(subset=["ExitSpeed"]).sort_values("Date")
        if len(batted_sorted) >= 5:
            w = min(15, len(batted_sorted))
            batted_sorted = batted_sorted.copy()
            batted_sorted["Roll"] = batted_sorted["ExitSpeed"].rolling(w, min_periods=3).mean()
            db_avg = all_stats["AvgEV"].mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(batted_sorted))), y=batted_sorted["Roll"],
                mode="lines", line=dict(color="#e63946", width=2.5), showlegend=False,
                hovertemplate="BIP #%{x}<br>Rolling EV: %{y:.1f} mph<extra></extra>",
            ))
            fig.add_hline(y=db_avg, line_dash="dash", line_color="#9e9e9e",
                          annotation_text=f"DB Avg: {db_avg:.1f}",
                          annotation_position="bottom right",
                          annotation_font=dict(size=10, color="#666"))
            fig.update_layout(xaxis_title="Batted Ball #", yaxis_title="EV (mph)",
                              height=300, **CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True, key="hitter_roll_ev")
        else:
            st.info("Not enough batted balls for rolling chart.")

    with col6:
        section_header("Exit Velo vs Launch Angle")
        ev_la = in_play.dropna(subset=["ExitSpeed", "Angle"])
        if not ev_la.empty:
            fig = px.scatter(
                ev_la, x="Angle", y="ExitSpeed", color="TaggedHitType",
                color_discrete_map={"GroundBall": "#d62728", "LineDrive": "#2ca02c",
                                    "FlyBall": "#1f77b4", "Popup": "#ff7f0e"},
                opacity=0.7, labels={"Angle": "Launch Angle", "ExitSpeed": "Exit Velo"},
            )
            # Barrel zone — trace actual curved boundary
            _bz_ev = np.linspace(98, max(ev_la["ExitSpeed"].max() + 2, 105), 40)
            _bz_la_lo = np.clip(26 - 2 * (_bz_ev - 98), 8, 26)
            _bz_la_hi = np.clip(30 + 3 * (_bz_ev - 98), 30, 50)
            fig.add_trace(go.Scatter(
                x=np.concatenate([_bz_la_lo, _bz_la_hi[::-1]]),
                y=np.concatenate([_bz_ev, _bz_ev[::-1]]),
                fill="toself", fillcolor="rgba(230,57,70,0.06)",
                line=dict(color="#e63946", width=1.5, dash="dash"),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_annotation(x=20, y=_bz_ev[-1], text="BARREL ZONE",
                               showarrow=False, font=dict(size=9, color="#e63946"))
            fig.update_layout(height=300, showlegend=True,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                          font=dict(size=9, color="#1a1a2e")),
                              **CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True, key="hitter_ev_la")
        else:
            st.info("No exit velo / launch angle data.")

    with col7:
        section_header("Swing Decisions (Heatmap)")
        loc = bdf.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        swing_loc = loc[loc["PitchCall"].isin(SWING_CALLS)]
        if not swing_loc.empty:
            fig = px.density_heatmap(
                swing_loc, x="PlateLocSide", y="PlateLocHeight",
                nbinsx=14, nbinsy=14, color_continuous_scale="YlOrRd",
            )
            add_strike_zone(fig)
            fig.update_layout(
                xaxis=dict(range=[-2.5, 2.5], scaleanchor="y", showticklabels=False, title=""),
                yaxis=dict(range=[0, 5], showticklabels=False, title=""),
                height=300, margin=dict(l=0, r=0, t=5, b=0),
                coloraxis_showscale=False, plot_bgcolor="white", paper_bgcolor="white",
                font=dict(color="#1a1a2e", family="Inter, Arial, sans-serif"),
            )
            st.plotly_chart(fig, use_container_width=True, key="hitter_swing_hm")
        else:
            st.info("No swing location data.")

    # ── ROW 4: Pitch Tracking by Type ──
    st.markdown("---")
    section_header("Pitch Tracking (Performance vs Pitch Types)")
    pitch_groups = {
        "Fastball": ["Fastball", "Sinker", "Cutter"],
        "Breaking": ["Slider", "Curveball", "Sweeper", "Knuckle Curve"],
        "Offspeed": ["Changeup", "Splitter"],
    }

    pt_rows = []
    for group_name, types in pitch_groups.items():
        sub = bdf[bdf["TaggedPitchType"].isin(types)]
        if len(sub) < 3:
            continue
        sub_swings = sub[sub["PitchCall"].isin(SWING_CALLS)]
        sub_whiffs = sub[sub["PitchCall"] == "StrikeSwinging"]
        sub_ip = sub[sub["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
        _pa_cols = ["GameID", "Inning", "PAofInning", "Batter"]
        if all(c in sub.columns for c in _pa_cols):
            sub_pa = sub.drop_duplicates(subset=_pa_cols).shape[0]
        else:
            sub_pa = int(sub["PitchofPA"].eq(1).sum())
        sub_ks = len(sub[sub["KorBB"] == "Strikeout"].drop_duplicates(subset=_pa_cols)) if all(c in sub.columns for c in _pa_cols) else len(sub[sub["KorBB"] == "Strikeout"])
        pt_rows.append({
            "Pitch Group": group_name,
            "#": len(sub),
            "%": round(len(sub) / len(bdf) * 100, 1),
            "PA": sub_pa,
            "SO": sub_ks,
            "BBE": len(sub_ip),
            "EV": round(sub_ip["ExitSpeed"].mean(), 1) if len(sub_ip) > 0 else None,
            "LA": round(sub_ip["Angle"].mean(), 1) if len(sub_ip) > 0 and sub_ip["Angle"].notna().any() else None,
            "Whiff%": round(len(sub_whiffs) / max(len(sub_swings), 1) * 100, 1),
        })
    if pt_rows:
        st.dataframe(pd.DataFrame(pt_rows), use_container_width=True, hide_index=True)

    # Per-pitch type breakdown (only main pitches)
    bdf_filtered = filter_minor_pitches(bdf)
    pt_detail_rows = []
    for pt in sorted(bdf_filtered["TaggedPitchType"].dropna().unique()):
        sub = bdf_filtered[bdf_filtered["TaggedPitchType"] == pt]
        if len(sub) < 3:
            continue
        sub_swings = sub[sub["PitchCall"].isin(SWING_CALLS)]
        sub_whiffs = sub[sub["PitchCall"] == "StrikeSwinging"]
        sub_ip = sub[sub["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
        pt_detail_rows.append({
            "Pitch": pt,
            "#": len(sub),
            "%": round(len(sub) / len(bdf) * 100, 1),
            "BBE": len(sub_ip),
            "EV": round(sub_ip["ExitSpeed"].mean(), 1) if len(sub_ip) > 0 else None,
            "LA": round(sub_ip["Angle"].mean(), 1) if len(sub_ip) > 0 and sub_ip["Angle"].notna().any() else None,
            "Whiff%": round(len(sub_whiffs) / max(len(sub_swings), 1) * 100, 1),
        })
    if pt_detail_rows:
        with st.expander("Detailed Pitch Type Breakdown"):
            st.dataframe(pd.DataFrame(pt_detail_rows), use_container_width=True, hide_index=True)

    # ── COUNT-BASED PERFORMANCE (Visual Heatmap Grid) ──
    st.markdown("---")
    section_header("Count-Based Performance")
    if "Balls" in bdf.columns and "Strikes" in bdf.columns:
        # Compute DB-wide averages per count for context
        all_data_counts = data.dropna(subset=["Balls", "Strikes"]).copy()
        all_data_counts["Count"] = all_data_counts["Balls"].astype(int).astype(str) + "-" + all_data_counts["Strikes"].astype(int).astype(str)
        bdf_counts = bdf.dropna(subset=["Balls", "Strikes"]).copy()
        bdf_counts["Count"] = bdf_counts["Balls"].astype(int).astype(str) + "-" + bdf_counts["Strikes"].astype(int).astype(str)

        def _count_stats(sub):
            sw = sub[sub["PitchCall"].isin(SWING_CALLS)]
            wh = sub[sub["PitchCall"] == "StrikeSwinging"]
            ip = sub[sub["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            return {
                "n": len(sub), "swings": len(sw),
                "swing_pct": len(sw) / max(len(sub), 1) * 100,
                "whiff_pct": len(wh) / max(len(sw), 1) * 100 if len(sw) > 0 else 0,
                "ev": ip["ExitSpeed"].mean() if len(ip) > 0 else np.nan,
                "bbe": len(ip),
            }

        # DB averages per count
        db_count_avgs = {}
        for cnt, grp in all_data_counts.groupby("Count"):
            db_count_avgs[cnt] = _count_stats(grp)

        # Player stats per count
        player_count_stats = {}
        for cnt, grp in bdf_counts.groupby("Count"):
            player_count_stats[cnt] = _count_stats(grp)

        # Visual count grid — 4 balls x 3 strikes, color-coded by Whiff% vs DB avg
        st.markdown(
            '<p style="font-size:12px;color:#666;margin-bottom:4px;">'
            'Each cell shows this hitter\'s Whiff% in that count. '
            '<span style="color:#1a7a1a;font-weight:700;">Green = better than DB avg</span> (lower whiff), '
            '<span style="color:#cc0000;font-weight:700;">Red = worse</span> (higher whiff). '
            'Hover labels show the full context.</p>',
            unsafe_allow_html=True,
        )

        count_metric = st.radio("Count Grid Metric", ["Whiff%", "Swing%", "Avg EV"], horizontal=True, key="hc_count_metric")
        metric_key = {"Whiff%": "whiff_pct", "Swing%": "swing_pct", "Avg EV": "ev"}[count_metric]
        # For whiff and swing, lower is better for hitter; for EV, higher is better
        higher_better = count_metric == "Avg EV"

        # Build heatmap grid
        balls_range = [0, 1, 2, 3]
        strikes_range = [0, 1, 2]
        z_vals = []
        hover_text = []
        for s in strikes_range:
            row_z = []
            row_hover = []
            for b in balls_range:
                cnt = f"{b}-{s}"
                ps = player_count_stats.get(cnt, {})
                db = db_count_avgs.get(cnt, {})
                p_val = ps.get(metric_key, np.nan)
                db_val = db.get(metric_key, np.nan)
                n = ps.get("n", 0)
                if n < 3 or pd.isna(p_val):
                    row_z.append(np.nan)
                    row_hover.append(f"{cnt}<br>Not enough data")
                else:
                    # Compute difference vs DB (positive = better for hitter)
                    if higher_better:
                        diff = p_val - db_val if not pd.isna(db_val) else 0
                    else:
                        diff = db_val - p_val if not pd.isna(db_val) else 0
                    row_z.append(diff)
                    db_str = f"{db_val:.1f}" if not pd.isna(db_val) else "N/A"
                    verdict = "BETTER" if diff > 0 else ("WORSE" if diff < 0 else "AVERAGE")
                    v_color = "#1a7a1a" if diff > 0 else ("#cc0000" if diff < 0 else "#666")
                    row_hover.append(
                        f"<b>{cnt}</b> ({n} pitches)<br>"
                        f"{count_metric}: <b>{p_val:.1f}{'%' if metric_key != 'ev' else ' mph'}</b><br>"
                        f"DB Avg: {db_str}{'%' if metric_key != 'ev' else ' mph'}<br>"
                        f"<span style='color:{v_color}'>{verdict} ({diff:+.1f})</span>"
                    )
            z_vals.append(row_z)
            hover_text.append(row_hover)

        # Build annotations with actual values
        annotations = []
        for si, s in enumerate(strikes_range):
            for bi, b in enumerate(balls_range):
                cnt = f"{b}-{s}"
                ps = player_count_stats.get(cnt, {})
                db = db_count_avgs.get(cnt, {})
                p_val = ps.get(metric_key, np.nan)
                db_val = db.get(metric_key, np.nan)
                n = ps.get("n", 0)
                if n >= 3 and not pd.isna(p_val):
                    annotations.append(dict(
                        x=bi, y=si, text=f"<b>{p_val:.1f}</b><br><span style='font-size:9px'>DB: {db_val:.1f}</span>" if not pd.isna(db_val) else f"<b>{p_val:.1f}</b>",
                        showarrow=False, font=dict(size=13, color="white", family="Inter"),
                    ))
                else:
                    annotations.append(dict(
                        x=bi, y=si, text="—", showarrow=False,
                        font=dict(size=13, color="#aaa", family="Inter"),
                    ))

        fig = go.Figure(data=go.Heatmap(
            z=z_vals,
            x=[f"{b} Balls" for b in balls_range],
            y=[f"{s} Strikes" for s in strikes_range],
            colorscale=[
                [0.0, "#cc0000"], [0.35, "#e8a0a0"], [0.5, "#f0f0f0"],
                [0.65, "#a0d4a0"], [1.0, "#1a7a1a"],
            ],
            zmid=0, showscale=False,
            hovertext=hover_text, hoverinfo="text",
            xgap=3, ygap=3,
        ))
        fig.update_layout(
            annotations=annotations,
            xaxis=dict(side="top", tickfont=dict(size=12, color="#1a1a2e")),
            yaxis=dict(tickfont=dict(size=12, color="#1a1a2e"), autorange="reversed"),
            height=250, margin=dict(l=80, r=20, t=40, b=20),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Inter, Arial, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True, key="hitter_count_grid")

        # Count-state summary with percentile bars
        st.markdown("")
        bdf_st = bdf.dropna(subset=["Balls", "Strikes"]).copy()
        bdf_st["CountState"] = "Even"
        bdf_st.loc[bdf_st["Strikes"] > bdf_st["Balls"], "CountState"] = "Behind"
        bdf_st.loc[bdf_st["Balls"] > bdf_st["Strikes"], "CountState"] = "Ahead"
        all_st = data.dropna(subset=["Balls", "Strikes"]).copy()
        all_st["CountState"] = "Even"
        all_st.loc[all_st["Strikes"] > all_st["Balls"], "CountState"] = "Behind"
        all_st.loc[all_st["Balls"] > all_st["Strikes"], "CountState"] = "Ahead"

        # Compute per-batter stats by count state for percentiles
        def _batter_count_state_stats(full_data, state):
            sub = full_data[full_data["CountState"] == state] if state else full_data
            rows = []
            for batter, grp in sub.groupby("Batter"):
                if len(grp) < 10:
                    continue
                sw = grp[grp["PitchCall"].isin(SWING_CALLS)]
                wh = grp[grp["PitchCall"] == "StrikeSwinging"]
                ip = grp[grp["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
                rows.append({
                    "Batter": batter,
                    "whiff_pct": len(wh) / max(len(sw), 1) * 100 if len(sw) > 0 else np.nan,
                    "swing_pct": len(sw) / max(len(grp), 1) * 100,
                    "ev": ip["ExitSpeed"].mean() if len(ip) > 0 else np.nan,
                })
            return pd.DataFrame(rows)

        state_metrics = []
        for state in ["Ahead", "Even", "Behind"]:
            p_sub = bdf_st[bdf_st["CountState"] == state]
            if len(p_sub) < 5:
                continue
            p_stats = _count_stats(p_sub)
            all_state_df = _batter_count_state_stats(data.dropna(subset=["Balls", "Strikes"]).assign(
                CountState=lambda x: np.where(x["Strikes"] > x["Balls"], "Behind",
                                    np.where(x["Balls"] > x["Strikes"], "Ahead", "Even"))
            ), state)
            whiff_pct = get_percentile(p_stats["whiff_pct"], all_state_df["whiff_pct"]) if not all_state_df.empty else 50
            ev_pct = get_percentile(p_stats["ev"], all_state_df["ev"]) if not all_state_df.empty and not pd.isna(p_stats["ev"]) else np.nan
            state_metrics.append(
                (f"{state} Whiff%", p_stats["whiff_pct"], whiff_pct, ".1f", False)
            )
            if not pd.isna(ev_pct):
                state_metrics.append(
                    (f"{state} EV", p_stats["ev"], ev_pct, ".1f", True)
                )

        if state_metrics:
            render_savant_percentile_section(state_metrics, "Count-State Performance (Percentile)")
            st.caption("Percentile vs. all batters in DB with 10+ pitches in that count state")

    else:
        st.caption("Count data not available.")

    # ── SITUATIONAL SPLITS (Percentile-Based Visual) ──
    st.markdown("---")
    section_header("Situational Splits")

    def _compute_split_percentile(player_df, all_data, split_col, split_val, filter_fn=None):
        """Compute player stats and percentile rank for a split."""
        if filter_fn:
            p_sub = filter_fn(player_df)
            a_sub = filter_fn(all_data)
        else:
            p_sub = player_df[player_df[split_col] == split_val] if split_col else player_df
            a_sub = all_data[all_data[split_col] == split_val] if split_col else all_data
        if len(p_sub) < 5:
            return None

        p_sw = p_sub[p_sub["PitchCall"].isin(SWING_CALLS)]
        p_wh = p_sub[p_sub["PitchCall"] == "StrikeSwinging"]
        p_ip = p_sub[p_sub["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])

        # All batters in this split
        batter_rows = []
        for batter, grp in a_sub.groupby("Batter"):
            if len(grp) < 10:
                continue
            sw = grp[grp["PitchCall"].isin(SWING_CALLS)]
            wh = grp[grp["PitchCall"] == "StrikeSwinging"]
            ip = grp[grp["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            batter_rows.append({
                "whiff": len(wh) / max(len(sw), 1) * 100 if len(sw) > 0 else np.nan,
                "ev": ip["ExitSpeed"].mean() if len(ip) > 0 else np.nan,
                "chase": (lambda _iz=in_zone_mask(grp): len(grp[(~_iz) & grp["PlateLocSide"].notna()][grp["PitchCall"].isin(SWING_CALLS)]) / max(len(grp[(~_iz) & grp["PlateLocSide"].notna()]), 1) * 100)() if grp["PlateLocSide"].notna().any() else np.nan,
            })
        all_df = pd.DataFrame(batter_rows)

        p_whiff = len(p_wh) / max(len(p_sw), 1) * 100 if len(p_sw) > 0 else np.nan
        p_ev = p_ip["ExitSpeed"].mean() if len(p_ip) > 0 else np.nan

        return {
            "n": len(p_sub), "bbe": len(p_ip),
            "whiff": p_whiff,
            "whiff_pct": get_percentile(p_whiff, all_df["whiff"]) if not all_df.empty and not pd.isna(p_whiff) else np.nan,
            "ev": p_ev,
            "ev_pct": get_percentile(p_ev, all_df["ev"]) if not all_df.empty and not pd.isna(p_ev) else np.nan,
        }

    sit_metrics = []

    # vs RHP / LHP
    if "PitcherThrows" in bdf.columns:
        for side, label in [("Right", "vs RHP"), ("Left", "vs LHP")]:
            res = _compute_split_percentile(bdf, data, "PitcherThrows", side)
            if res:
                sit_metrics.append((f"{label} Whiff%", res["whiff"], res["whiff_pct"], ".1f", False))
                if not pd.isna(res["ev_pct"]):
                    sit_metrics.append((f"{label} Avg EV", res["ev"], res["ev_pct"], ".1f", True))

    # By inning group
    if "Inning" in bdf.columns:
        for label, lo, hi in [("Early Inn.", 1, 3), ("Mid Inn.", 4, 6), ("Late Inn.", 7, 20)]:
            res = _compute_split_percentile(bdf, data, None, None,
                filter_fn=lambda df, l=lo, h=hi: df[df["Inning"].between(l, h)])
            if res:
                sit_metrics.append((f"{label} Whiff%", res["whiff"], res["whiff_pct"], ".1f", False))
                if not pd.isna(res["ev_pct"]):
                    sit_metrics.append((f"{label} Avg EV", res["ev"], res["ev_pct"], ".1f", True))

    if sit_metrics:
        render_savant_percentile_section(sit_metrics, None)
        st.caption("Percentile vs. all batters in DB (min 10 pitches in that split)")


# ──────────────────────────────────────────────
# SWING DECISION LAB
# ──────────────────────────────────────────────

def _swing_decision_lab(data, batter, season_filter, bdf, batted, pr, all_batter_stats):
    """Render the Swing Decision Lab — plate discipline, decision maps,
    pitch-type breakdowns, count-state analysis, and recommendations."""

    all_stats = all_batter_stats
    # Use adaptive per-batter zone (matches population computation)
    batter_zones = _build_batter_zones(data)
    iz = in_zone_mask(bdf, batter_zones, batter_col="Batter")
    oz = ~iz & bdf["PlateLocSide"].notna() & bdf["PlateLocHeight"].notna()
    swings = bdf[bdf["PitchCall"].isin(SWING_CALLS)]
    whiffs = bdf[bdf["PitchCall"] == "StrikeSwinging"]
    contacts = bdf[bdf["PitchCall"].isin(CONTACT_CALLS)]
    in_zone_pitches = bdf[iz]
    out_zone_pitches = bdf[oz]
    swings_iz = in_zone_pitches[in_zone_pitches["PitchCall"].isin(SWING_CALLS)]
    swings_oz = out_zone_pitches[out_zone_pitches["PitchCall"].isin(SWING_CALLS)]
    contacts_iz = in_zone_pitches[in_zone_pitches["PitchCall"].isin(CONTACT_CALLS)]
    contacts_oz = out_zone_pitches[out_zone_pitches["PitchCall"].isin(CONTACT_CALLS)]

    zone_swing_pct = len(swings_iz) / max(len(in_zone_pitches), 1) * 100
    chase_pct = len(swings_oz) / max(len(out_zone_pitches), 1) * 100
    whiff_pct = len(whiffs) / max(len(swings), 1) * 100
    zone_contact_pct = len(contacts_iz) / max(len(swings_iz), 1) * 100 if len(swings_iz) > 0 else np.nan
    chase_contact_pct = len(contacts_oz) / max(len(swings_oz), 1) * 100 if len(swings_oz) > 0 else np.nan

    # ═══════════════════════════════════════════
    # SECTION A: Decision Profile Summary
    # ═══════════════════════════════════════════
    section_header("Decision Profile")
    st.caption("Plate discipline metrics vs. all hitters in the database.")

    disc_metrics = [
        ("Zone Swing%", zone_swing_pct, get_percentile(zone_swing_pct, all_stats["ZoneSwingPct"]), ".1f", True),
        ("Chase%", chase_pct, get_percentile(chase_pct, all_stats["ChasePct"]), ".1f", False),
        ("Whiff%", whiff_pct, get_percentile(whiff_pct, all_stats["WhiffPct"]), ".1f", False),
        ("Z-Contact%", zone_contact_pct, get_percentile(zone_contact_pct, all_stats["ZoneContactPct"]), ".1f", True),
        ("Chase Contact%", chase_contact_pct,
         get_percentile(chase_contact_pct, all_stats["ChaseContact"]) if not pd.isna(chase_contact_pct) else np.nan,
         ".1f", True),
    ]
    render_savant_percentile_section(disc_metrics)

    # Decision Score composite
    zs_pctl = get_percentile(zone_swing_pct, all_stats["ZoneSwingPct"])
    ch_pctl = get_percentile(chase_pct, all_stats["ChasePct"])
    zc_pctl = get_percentile(zone_contact_pct, all_stats["ZoneContactPct"]) if not pd.isna(zone_contact_pct) else 50
    # For chase, lower is better → invert percentile
    ch_pctl_good = (100 - ch_pctl) if not pd.isna(ch_pctl) else 50
    zs_pctl_safe = zs_pctl if not pd.isna(zs_pctl) else 50
    zc_pctl_safe = zc_pctl if not pd.isna(zc_pctl) else 50
    decision_score = round(zs_pctl_safe * 0.35 + ch_pctl_good * 0.40 + zc_pctl_safe * 0.25, 0)

    # Verdict — nuanced based on the specific metric profile
    high_chase = chase_pct > 32 if not pd.isna(chase_pct) else False
    low_chase = chase_pct < 25 if not pd.isna(chase_pct) else False
    passive_zone = zone_swing_pct < 60
    aggressive_zone = zone_swing_pct > 72
    high_zc = zone_contact_pct > 85 if not pd.isna(zone_contact_pct) else False
    low_zc = zone_contact_pct < 75 if not pd.isna(zone_contact_pct) else False

    if decision_score >= 80:
        if low_chase and aggressive_zone:
            verdict = "Elite plate discipline — hunts strikes and lays off everything else"
        elif low_chase and high_zc:
            verdict = "Exceptional pitch recognition with elite zone contact"
        else:
            verdict = "Top-tier decision-maker — controls the at-bat consistently"
    elif decision_score >= 65:
        if high_chase:
            verdict = f"Strong zone attacker but chases too much ({chase_pct:.0f}% chase rate)"
        elif passive_zone:
            verdict = f"Great eye but could be more aggressive on hittable pitches ({zone_swing_pct:.0f}% zone swing)"
        elif low_zc:
            verdict = "Good pitch selection, needs to barrel more pitches in the zone"
        else:
            verdict = "Above-average approach — makes pitchers work for outs"
    elif decision_score >= 50:
        if high_chase and passive_zone:
            verdict = "Swings at the wrong pitches — chases out of zone, takes in zone"
        elif high_chase:
            verdict = f"Expanding the zone too often ({chase_pct:.0f}% chase) — hurting the at-bat"
        elif passive_zone:
            verdict = f"Leaving hittable pitches in the zone ({zone_swing_pct:.0f}% zone swing)"
        else:
            verdict = "Middle-of-the-pack approach — no glaring weakness but no strength either"
    elif decision_score >= 35:
        if high_chase and low_zc:
            verdict = "Chasing off the plate and struggling to connect in the zone"
        elif aggressive_zone and high_chase:
            verdict = f"Free swinger — aggressive everywhere, {chase_pct:.0f}% chase rate"
        elif passive_zone:
            verdict = "Too passive — watching too many hittable pitches go by"
        else:
            verdict = "Below-average pitch selection — approach needs refinement"
    else:
        if high_chase and passive_zone:
            verdict = "Inverted approach — chasing balls while taking strikes"
        elif high_chase:
            verdict = f"Expands aggressively ({chase_pct:.0f}% chase) — pitchers exploiting the zone"
        else:
            verdict = "Significant swing-decision issues — needs mechanical or recognition work"

    ds_color = "#22c55e" if decision_score >= 70 else "#3b82f6" if decision_score >= 55 else "#f59e0b" if decision_score >= 40 else "#ef4444"
    st.markdown(
        f'<div style="padding:10px 16px;border-radius:8px;border-left:5px solid {ds_color};'
        f'background:{ds_color}10;margin:8px 0;">'
        f'<span style="font-size:20px;font-weight:bold;color:{ds_color};">Decision Score: {decision_score:.0f}</span>'
        f'<span style="font-size:13px;color:#555;margin-left:12px;">{verdict}</span>'
        f'</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    # SECTION B: Swing Decision Zone Maps
    # ═══════════════════════════════════════════
    st.markdown("---")
    section_header("Swing Decision Zone Maps")
    st.caption("Where should this hitter swing vs. where they actually swing? Mismatch = coaching opportunity.")

    loc_data = bdf.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
    _bs = safe_mode(bdf["BatterSide"], "Right") if "BatterSide" in bdf.columns else "Right"
    is_rhh = _bs != "Left"
    if len(loc_data) >= 30:
        # Build 5x5 grid
        h_edges = np.linspace(-1.5, 1.5, 6)
        v_edges = np.linspace(0.5, 4.5, 6)

        should_swing = np.full((5, 5), np.nan)
        actually_swings = np.full((5, 5), np.nan)
        mismatch = np.full((5, 5), np.nan)

        for vi in range(5):
            for hi in range(5):
                mask = (
                    (loc_data["PlateLocSide"] >= h_edges[hi]) &
                    (loc_data["PlateLocSide"] < h_edges[hi + 1]) &
                    (loc_data["PlateLocHeight"] >= v_edges[vi]) &
                    (loc_data["PlateLocHeight"] < v_edges[vi + 1])
                )
                cell = loc_data[mask]
                if len(cell) < 3:
                    continue
                cell_swings = cell[cell["PitchCall"].isin(SWING_CALLS)]
                swing_rate = len(cell_swings) / len(cell) * 100
                actually_swings[vi, hi] = swing_rate

                cell_ip = cell[(cell["PitchCall"] == "InPlay") & cell["ExitSpeed"].notna()]
                cell_whiffs = cell[cell["PitchCall"] == "StrikeSwinging"]
                if len(cell_swings) >= 2:
                    contact_rate = 1 - len(cell_whiffs) / len(cell_swings) if len(cell_swings) > 0 else 0
                    avg_ev = cell_ip["ExitSpeed"].mean() if len(cell_ip) >= 2 else 70
                    ev_score = min(max((avg_ev - 70) / 30 * 100, 0), 100) if not pd.isna(avg_ev) else 30
                    contact_score = contact_rate * 100
                    should_score = ev_score * 0.6 + contact_score * 0.4
                    should_swing[vi, hi] = should_score
                    mismatch[vi, hi] = swing_rate - should_score

        # Label columns relative to batter handedness.
        # Trackman: col 0 = most negative PlateLocSide = 3B side.
        # RHH: 3B side = inside → labels left-to-right: In … Away
        # LHH: 3B side = away  → labels left-to-right: Away … In
        if is_rhh:
            h_labels = ["In", "In-Mid", "Mid", "Away-Mid", "Away"]
        else:
            h_labels = ["Away", "Away-Mid", "Mid", "In-Mid", "In"]
        v_labels = ["Low", "Low-Mid", "Mid", "Up-Mid", "Up"]

        map1, map2, map3 = st.columns(3)
        with map1:
            st.markdown("**Should Swing**")
            st.caption("Green = good outcomes when swinging here")
            fig_should = px.imshow(
                np.flipud(should_swing), text_auto=".0f",
                color_continuous_scale=[[0, "#ef4444"], [0.5, "#fbbf24"], [1, "#22c55e"]],
                x=h_labels, y=list(reversed(v_labels)),
                labels=dict(color="Score"), aspect="auto",
            )
            _add_grid_zone_outline(fig_should)
            fig_should.update_layout(height=320, coloraxis_showscale=False, **CHART_LAYOUT)
            st.plotly_chart(fig_should, use_container_width=True, key="sdl_should")

        with map2:
            st.markdown("**Actually Swings**")
            st.caption("Swing rate by zone cell")
            fig_actual = px.imshow(
                np.flipud(actually_swings), text_auto=".0f",
                color_continuous_scale="YlOrRd",
                x=h_labels, y=list(reversed(v_labels)),
                labels=dict(color="Swing%"), aspect="auto",
            )
            _add_grid_zone_outline(fig_actual)
            fig_actual.update_layout(height=320, coloraxis_showscale=False, **CHART_LAYOUT)
            st.plotly_chart(fig_actual, use_container_width=True, key="sdl_actual")

        with map3:
            st.markdown("**Mismatch**")
            st.caption("Red = swings too much, Blue = should swing more")
            mm_rounded = np.round(mismatch)
            mm_text = []
            for row in np.flipud(mm_rounded):
                row_text = []
                for v in row:
                    row_text.append(f"{v:+.0f}" if not np.isnan(v) else "")
                mm_text.append(row_text)
            fig_mm = px.imshow(
                np.flipud(mm_rounded),
                color_continuous_scale=[[0, "#3b82f6"], [0.5, "white"], [1, "#ef4444"]],
                x=h_labels, y=list(reversed(v_labels)),
                labels=dict(color="Over/Under"), aspect="auto",
                zmin=-50, zmax=50,
            )
            fig_mm.update_traces(text=mm_text, texttemplate="%{text}")
            _add_grid_zone_outline(fig_mm)
            fig_mm.update_layout(height=320, coloraxis_showscale=False, **CHART_LAYOUT)
            st.plotly_chart(fig_mm, use_container_width=True, key="sdl_mismatch")
    else:
        st.info("Not enough location data for zone maps (need 30+ pitches).")

    # ═══════════════════════════════════════════
    # SECTION C: Pitch-Type Decision Breakdown
    # ═══════════════════════════════════════════
    st.markdown("---")
    section_header("Pitch-Type Decision Breakdown")
    st.caption("How does this hitter decide against each pitch type? In-zone vs. out-of-zone splits.")

    pitch_types_seen = sorted(bdf["TaggedPitchType"].dropna().unique())
    # Compute team averages for comparison
    all_dav_bat = filter_davidson(data, "batter")
    all_dav_bat = all_dav_bat[all_dav_bat["Season"].isin(season_filter)]
    team_batter_zones = _build_batter_zones(all_dav_bat)

    pt_rows = []
    flags = []
    for pt in pitch_types_seen:
        pt_d = bdf[bdf["TaggedPitchType"] == pt]
        if len(pt_d) < 5:
            continue
        pt_iz = pt_d[in_zone_mask(pt_d, batter_zones, "Batter")]
        pt_oz_mask = ~in_zone_mask(pt_d, batter_zones, "Batter") & pt_d["PlateLocSide"].notna()
        pt_oz = pt_d[pt_oz_mask]
        pt_sw = pt_d[pt_d["PitchCall"].isin(SWING_CALLS)]
        pt_wh = pt_d[pt_d["PitchCall"] == "StrikeSwinging"]
        pt_sw_oz = pt_oz[pt_oz["PitchCall"].isin(SWING_CALLS)]
        pt_sw_iz = pt_iz[pt_iz["PitchCall"].isin(SWING_CALLS)]
        pt_ip = pt_d[(pt_d["PitchCall"] == "InPlay") & pt_d["ExitSpeed"].notna()]

        swing_pct = len(pt_sw) / max(len(pt_d), 1) * 100
        whiff_pt = len(pt_wh) / max(len(pt_sw), 1) * 100 if len(pt_sw) > 0 else 0
        chase_pt = len(pt_sw_oz) / max(len(pt_oz), 1) * 100 if len(pt_oz) > 0 else 0
        zone_swing_pt = len(pt_sw_iz) / max(len(pt_iz), 1) * 100 if len(pt_iz) > 0 else 0
        take_iz_pct = 100 - zone_swing_pt
        avg_ev_pt = pt_ip["ExitSpeed"].mean() if len(pt_ip) >= 2 else np.nan

        pt_rows.append({
            "Pitch": pt, "Seen": len(pt_d),
            "Swing%": round(swing_pct, 1),
            "Whiff%": round(whiff_pt, 1),
            "Chase%": round(chase_pt, 1),
            "Z-Swing%": round(zone_swing_pt, 1),
            "Z-Take%": round(take_iz_pct, 1),
            "Avg EV": round(avg_ev_pt, 1) if not pd.isna(avg_ev_pt) else np.nan,
        })

        # Compute team averages for this pitch type to flag outliers
        team_pt = all_dav_bat[all_dav_bat["TaggedPitchType"] == pt]
        if len(team_pt) >= 20:
            team_oz_mask = ~in_zone_mask(team_pt, team_batter_zones, "Batter") & team_pt["PlateLocSide"].notna()
            team_oz = team_pt[team_oz_mask]
            team_sw_oz = team_oz[team_oz["PitchCall"].isin(SWING_CALLS)]
            team_chase = len(team_sw_oz) / max(len(team_oz), 1) * 100

            team_iz = team_pt[in_zone_mask(team_pt, team_batter_zones, "Batter")]
            team_sw_iz = team_iz[team_iz["PitchCall"].isin(SWING_CALLS)]
            team_z_swing = len(team_sw_iz) / max(len(team_iz), 1) * 100

            if chase_pt > team_chase + 10 and len(pt_oz) >= 5:
                flags.append(f"Chases **{pt}** {chase_pt:.0f}% of the time (team avg: {team_chase:.0f}%)")
            if take_iz_pct > 30 and take_iz_pct > (100 - team_z_swing) + 10 and len(pt_iz) >= 5:
                ev_str = f", avg EV when swinging: {avg_ev_pt:.0f} mph" if not pd.isna(avg_ev_pt) else ""
                flags.append(f"Takes in-zone **{pt}** {take_iz_pct:.0f}% of the time — missing hittable pitches{ev_str}")

    if pt_rows:
        pt_df = pd.DataFrame(pt_rows)
        st.dataframe(pt_df, use_container_width=True, hide_index=True)

    if flags:
        st.markdown("**Flags:**")
        for f in flags:
            st.markdown(
                f'<div style="padding:4px 12px;margin:2px 0;font-size:12px;'
                f'background:#fff8e1;border-radius:4px;border-left:3px solid #f59e0b;">'
                f'{f}</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    # SECTION D: Count-State Decisions
    # ═══════════════════════════════════════════
    st.markdown("---")
    section_header("Count-State Decisions")

    # Ahead vs Behind metrics
    ahead_counts = [("1", "0"), ("2", "0"), ("2", "1"), ("3", "0"), ("3", "1")]
    behind_counts = [("0", "1"), ("0", "2"), ("1", "2")]

    def _count_metrics(count_list, label):
        frames = []
        for b, s in count_list:
            cd = bdf[(bdf["Balls"].astype(str) == b) & (bdf["Strikes"].astype(str) == s)]
            frames.append(cd)
        combined = pd.concat(frames) if frames else pd.DataFrame()
        if len(combined) < 5:
            return None
        c_sw = combined[combined["PitchCall"].isin(SWING_CALLS)]
        c_wh = combined[combined["PitchCall"] == "StrikeSwinging"]
        c_iz = combined[in_zone_mask(combined, batter_zones, "Batter")]
        c_oz_mask = ~in_zone_mask(combined, batter_zones, "Batter") & combined["PlateLocSide"].notna()
        c_oz = combined[c_oz_mask]
        c_sw_oz = c_oz[c_oz["PitchCall"].isin(SWING_CALLS)]
        c_ip = combined[(combined["PitchCall"] == "InPlay") & combined["ExitSpeed"].notna()]
        return {
            "label": label, "n": len(combined),
            "swing_pct": len(c_sw) / max(len(combined), 1) * 100,
            "chase_pct": len(c_sw_oz) / max(len(c_oz), 1) * 100 if len(c_oz) > 0 else 0,
            "whiff_pct": len(c_wh) / max(len(c_sw), 1) * 100 if len(c_sw) > 0 else 0,
            "avg_ev": c_ip["ExitSpeed"].mean() if len(c_ip) >= 2 else np.nan,
        }

    ahead_m = _count_metrics(ahead_counts, "Ahead (1-0, 2-0, 2-1, 3-0, 3-1)")
    behind_m = _count_metrics(behind_counts, "Behind (0-1, 0-2, 1-2)")

    if ahead_m and behind_m:
        cc1, cc2 = st.columns(2)
        for col, m, color in [(cc1, ahead_m, "#22c55e"), (cc2, behind_m, "#ef4444")]:
            with col:
                ev_str = f"{m['avg_ev']:.1f} mph" if not pd.isna(m['avg_ev']) else "-"
                st.markdown(
                    f'<div style="padding:10px 14px;border-radius:8px;border-left:4px solid {color};'
                    f'background:{color}10;margin:4px 0;">'
                    f'<div style="font-size:14px;font-weight:bold;">{m["label"]}</div>'
                    f'<div style="font-size:12px;color:#555;margin-top:4px;">'
                    f'Swing% <b>{m["swing_pct"]:.0f}</b> &middot; '
                    f'Chase% <b>{m["chase_pct"]:.0f}</b> &middot; '
                    f'Whiff% <b>{m["whiff_pct"]:.0f}</b> &middot; '
                    f'Avg EV <b>{ev_str}</b>'
                    f'</div><div style="font-size:11px;color:#888;">({m["n"]} pitches)</div>'
                    f'</div>', unsafe_allow_html=True)

        # Chase expansion: compare 0-0 vs 0-2 swing zones
        st.markdown("**Zone Expansion: 0-0 vs 0-2**")
        st.caption("How much does the swing zone expand when behind?")
        cnt_00 = bdf[(bdf["Balls"].astype(str) == "0") & (bdf["Strikes"].astype(str) == "0")]
        cnt_02 = bdf[(bdf["Balls"].astype(str) == "0") & (bdf["Strikes"].astype(str) == "2")]
        if len(cnt_00) >= 10 and len(cnt_02) >= 10:
            exp1, exp2 = st.columns(2)
            for col_exp, cnt_d, cnt_label in [(exp1, cnt_00, "0-0"), (exp2, cnt_02, "0-2")]:
                with col_exp:
                    cnt_loc = cnt_d.dropna(subset=["PlateLocSide", "PlateLocHeight"])
                    cnt_sw = cnt_loc[cnt_loc["PitchCall"].isin(SWING_CALLS)]
                    if len(cnt_sw) >= 3:
                        fig_exp = go.Figure(go.Histogram2d(
                            x=cnt_sw["PlateLocSide"], y=cnt_sw["PlateLocHeight"],
                            xbins=dict(start=-2.5, end=2.5, size=5.0/8),
                            ybins=dict(start=0, end=5, size=5.0/8),
                            colorscale="YlOrRd", showscale=False,
                        ))
                        add_strike_zone(fig_exp)
                        fig_exp.update_layout(
                            xaxis=dict(range=[-2.5, 2.5], title="", showticklabels=False,
                                       scaleanchor="y", scaleratio=1),
                            yaxis=dict(range=[0, 5], title="", showticklabels=False),
                            height=350, margin=dict(l=5, r=5, t=5, b=5),
                            plot_bgcolor="white", paper_bgcolor="white",
                        )
                        st.plotly_chart(fig_exp, use_container_width=True, key=f"sdl_exp_{cnt_label}")
                        # Compute chase rate for this count
                        cnt_iz = cnt_d[in_zone_mask(cnt_d, batter_zones, "Batter")]
                        cnt_oz_m = ~in_zone_mask(cnt_d, batter_zones, "Batter") & cnt_d["PlateLocSide"].notna()
                        cnt_oz = cnt_d[cnt_oz_m]
                        cnt_ch = len(cnt_oz[cnt_oz["PitchCall"].isin(SWING_CALLS)]) / max(len(cnt_oz), 1) * 100
                        st.caption(f"{cnt_label}: Chase {cnt_ch:.0f}% ({len(cnt_d)} pitches)")
                    else:
                        st.caption(f"{cnt_label}: not enough swings")
        else:
            st.info("Not enough data at 0-0 or 0-2 counts.")

    # Two-strike approach
    st.markdown("**Two-Strike Approach**")
    two_strike = bdf[pd.to_numeric(bdf["Strikes"], errors='coerce').fillna(0).astype(int) == 2]
    if len(two_strike) >= 10:
        ts_rows = []
        for pt in sorted(two_strike["TaggedPitchType"].dropna().unique()):
            pt_2s = two_strike[two_strike["TaggedPitchType"] == pt]
            if len(pt_2s) < 3:
                continue
            pt_2s_sw = pt_2s[pt_2s["PitchCall"].isin(SWING_CALLS)]
            pt_2s_wh = pt_2s[pt_2s["PitchCall"] == "StrikeSwinging"]
            pt_2s_oz_m = ~in_zone_mask(pt_2s, batter_zones, "Batter") & pt_2s["PlateLocSide"].notna()
            pt_2s_oz = pt_2s[pt_2s_oz_m]
            pt_2s_ch = pt_2s_oz[pt_2s_oz["PitchCall"].isin(SWING_CALLS)]
            pt_2s_ip = pt_2s[(pt_2s["PitchCall"] == "InPlay") & pt_2s["ExitSpeed"].notna()]
            ts_rows.append({
                "Pitch": pt, "N": len(pt_2s),
                "Whiff%": round(len(pt_2s_wh) / max(len(pt_2s_sw), 1) * 100, 1) if len(pt_2s_sw) > 0 else 0,
                "Chase%": round(len(pt_2s_ch) / max(len(pt_2s_oz), 1) * 100, 1) if len(pt_2s_oz) > 0 else 0,
                "Avg EV": round(pt_2s_ip["ExitSpeed"].mean(), 1) if len(pt_2s_ip) >= 2 else np.nan,
            })
        if ts_rows:
            ts_df = pd.DataFrame(ts_rows).sort_values("Whiff%", ascending=False)
            st.dataframe(ts_df, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough two-strike data.")

    # ═══════════════════════════════════════════
    # SECTION E: Swing Timing & Quality
    # ═══════════════════════════════════════════
    st.markdown("---")
    section_header("Swing Quality & Contact")

    # EV by zone location heatmap
    ip_loc = batted.dropna(subset=["PlateLocSide", "PlateLocHeight", "ExitSpeed"])
    if len(ip_loc) >= 15:
        st.markdown("**Exit Velocity by Location**")
        st.caption("Where does this hitter generate the most damage?")
        h_edges_ev = np.linspace(-1.5, 1.5, 6)
        v_edges_ev = np.linspace(0.5, 4.5, 6)
        ev_grid = np.full((5, 5), np.nan)
        for vi in range(5):
            for hi in range(5):
                mask = (
                    (ip_loc["PlateLocSide"] >= h_edges_ev[hi]) &
                    (ip_loc["PlateLocSide"] < h_edges_ev[hi + 1]) &
                    (ip_loc["PlateLocHeight"] >= v_edges_ev[vi]) &
                    (ip_loc["PlateLocHeight"] < v_edges_ev[vi + 1])
                )
                cell_ip = ip_loc[mask]
                if len(cell_ip) >= 2:
                    ev_grid[vi, hi] = cell_ip["ExitSpeed"].mean()
        if is_rhh:
            h_labels_ev = ["In", "In-Mid", "Mid", "Away-Mid", "Away"]
        else:
            h_labels_ev = ["Away", "Away-Mid", "Mid", "In-Mid", "In"]
        v_labels_ev = ["Low", "Low-Mid", "Mid", "Up-Mid", "Up"]
        fig_ev_zone = px.imshow(
            np.flipud(ev_grid), text_auto=".0f",
            color_continuous_scale=[[0, "#3b82f6"], [0.5, "white"], [1, "#ef4444"]],
            x=h_labels_ev, y=list(reversed(v_labels_ev)),
            labels=dict(color="Avg EV"), aspect="auto",
            zmin=70, zmax=100,
        )
        _add_grid_zone_outline(fig_ev_zone)
        fig_ev_zone.update_layout(height=350, **CHART_LAYOUT)
        st.plotly_chart(fig_ev_zone, use_container_width=True, key="sdl_ev_zone")

    # Bat speed proxy by pitch type (if enough hard hit data)
    hard_hit = batted[batted["ExitSpeed"] >= 80].copy()
    if len(hard_hit) >= 10 and "RelSpeed" in hard_hit.columns:
        hard_hit["BatSpeedProxy"] = (hard_hit["ExitSpeed"] - 0.2 * hard_hit["RelSpeed"]) / 1.2
        bs_by_pt = hard_hit.groupby("TaggedPitchType")["BatSpeedProxy"].agg(["mean", "count"]).reset_index()
        bs_by_pt = bs_by_pt[bs_by_pt["count"] >= 3].sort_values("mean", ascending=False)
        if not bs_by_pt.empty:
            st.markdown("**Estimated Bat Speed by Pitch Type**")
            st.caption("Higher = squaring up; lower = late/fooled. Based on EV >= 80 mph contact.")
            fig_bs = go.Figure()
            for _, row_bs in bs_by_pt.iterrows():
                pc = PITCH_COLORS.get(row_bs["TaggedPitchType"], "#888")
                fig_bs.add_trace(go.Bar(
                    x=[row_bs["TaggedPitchType"]], y=[row_bs["mean"]],
                    marker_color=pc, text=[f'{row_bs["mean"]:.1f}'],
                    textposition="outside", name=row_bs["TaggedPitchType"],
                    showlegend=False,
                ))
            fig_bs.update_layout(
                height=300, yaxis_title="Est. Bat Speed (mph)", **CHART_LAYOUT,
                yaxis=dict(range=[min(bs_by_pt["mean"].min() - 5, 55), bs_by_pt["mean"].max() + 5]),
            )
            st.plotly_chart(fig_bs, use_container_width=True, key="sdl_bat_speed")

    # ═══════════════════════════════════════════
    # SECTION F: Actionable Recommendations
    # ═══════════════════════════════════════════
    st.markdown("---")
    section_header("Recommendations")

    recs = []

    # Chase-related recommendations
    if not pd.isna(chase_pct):
        # Overall chase rate
        behind_chase = behind_m["chase_pct"] if behind_m else 0
        ahead_chase = ahead_m["chase_pct"] if ahead_m else 0
        if behind_m and ahead_m and behind_chase > ahead_chase + 12:
            recs.append(
                f"Tighten the zone with 2 strikes — chase rate jumps from "
                f"{ahead_chase:.0f}% when ahead to {behind_chase:.0f}% when behind")

    # Pitch-type specific chasing
    for f_text in flags:
        if "Chases" in f_text:
            recs.append(f_text.replace("**", ""))

    # Zone passivity
    if zone_swing_pct < 62:
        # Find the pitch type they take the most in-zone
        best_take_pt = None
        best_take_rate = 0
        for row_pt in pt_rows:
            if row_pt["Z-Take%"] > best_take_rate and row_pt["Seen"] >= 10:
                best_take_rate = row_pt["Z-Take%"]
                best_take_pt = row_pt["Pitch"]
        if best_take_pt and best_take_rate > 30:
            ev_val = next((r["Avg EV"] for r in pt_rows if r["Pitch"] == best_take_pt), np.nan)
            ev_str = f", avg EV when swinging: {ev_val:.0f} mph" if not pd.isna(ev_val) else ""
            recs.append(
                f"Attack the in-zone {best_take_pt} more — taking {best_take_rate:.0f}% of "
                f"hittable {best_take_pt}s{ev_str}")

    # Whiff-related
    if whiff_pct > 30:
        worst_whiff_pt = max(pt_rows, key=lambda r: r["Whiff%"]) if pt_rows else None
        if worst_whiff_pt and worst_whiff_pt["Whiff%"] > 35:
            recs.append(
                f"High whiff rate on {worst_whiff_pt['Pitch']} ({worst_whiff_pt['Whiff%']:.0f}%) "
                f"— consider shortening the swing or looking for it earlier in the count")

    if not recs:
        recs.append("No major red flags — continue current approach and focus on consistency")

    for i, rec in enumerate(recs[:4]):
        rec_color = "#f59e0b" if i < len(flags) else "#3b82f6"
        st.markdown(
            f'<div style="padding:6px 14px;margin:3px 0;font-size:13px;'
            f'border-radius:6px;border-left:4px solid {rec_color};background:{rec_color}08;">'
            f'{rec}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# PAGE: HITTING (merged Hitter Card + Hitters Lab)
# ──────────────────────────────────────────────
def page_hitting(data):
    hitting = filter_davidson(data, "batter")
    if hitting.empty:
        st.warning("No hitting data found.")
        return

    batters = sorted(hitting["Batter"].unique())
    c1, c2 = st.columns([2, 3])
    with c1:
        batter = st.selectbox("Select Hitter", batters, format_func=display_name, key="hitting_batter")
    with c2:
        all_seasons = get_all_seasons()
        season_filter = st.multiselect("Season", all_seasons, default=all_seasons, key="hitting_season")

    bdf = hitting[(hitting["Batter"] == batter) & (hitting["Season"].isin(season_filter))]
    if len(bdf) < 20:
        st.warning("Not enough pitches (need 20+) to analyze.")
        return

    all_batter_stats = compute_batter_stats_pop(season_filter=season_filter)
    if all_batter_stats.empty or batter not in all_batter_stats["Batter"].values:
        st.info("Not enough data for this player.")
        return
    pr = all_batter_stats[all_batter_stats["Batter"] == batter].iloc[0]

    batted = bdf[bdf["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])

    # Player header
    jersey = JERSEY.get(batter, "")
    pos = POSITION.get(batter, "")
    side = safe_mode(bdf["BatterSide"], "")
    bats = {"Right": "R", "Left": "L", "Switch": "S"}.get(side, side)
    player_header(batter, jersey, pos,
                  f"{pos}  |  Bats: {bats}  |  Davidson Wildcats",
                  f"{int(pr['PA'])} PA  |  {int(pr['BBE'])} Batted Balls  |  "
                  f"Seasons: {', '.join(str(int(s)) for s in sorted(season_filter))}")

    tab_card, tab_sdl = st.tabs(["Hitter Card", "Swing Decision Lab"])
    with tab_card:
        _hitter_card_content(data, batter, season_filter, bdf, batted, pr, all_batter_stats)
    with tab_sdl:
        _swing_decision_lab(data, batter, season_filter, bdf, batted, pr, all_batter_stats)


# ──────────────────────────────────────────────
# SAVANT MOVEMENT PROFILE (concentric circles)
# ──────────────────────────────────────────────
def make_movement_profile(pdf, height=520):
    """Create a Baseball Savant-style movement profile with concentric circles."""
    mov = pdf.dropna(subset=["HorzBreak", "InducedVertBreak"])
    if mov.empty:
        return None
    # Filter to main pitches only
    mov = filter_minor_pitches(mov)
    if mov.empty:
        return None

    fig = go.Figure()

    # Concentric circles at 6, 12, 18, 24 inches
    for r in [6, 12, 18, 24]:
        theta = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(go.Scatter(
            x=r * np.cos(theta), y=r * np.sin(theta), mode="lines",
            line=dict(color="#d0d8e0", width=1), showlegend=False, hoverinfo="skip",
        ))
        fig.add_annotation(x=0, y=r, text=f'{r}"', showarrow=False,
                           font=dict(size=9, color="#999"), yshift=8)

    # Crosshairs
    fig.add_hline(y=0, line_color="#bbb", line_width=1)
    fig.add_vline(x=0, line_color="#bbb", line_width=1)

    # Negate HorzBreak for Savant convention:
    # Trackman: positive HB = glove-side (toward 3B for RHP)
    # Savant chart: LEFT = toward 1B (arm-side), RIGHT = toward 3B (glove-side)
    # So we negate HB so that glove-side (positive trackman) plots LEFT (negative x)
    mov = mov.copy()
    mov["HB_plot"] = -mov["HorzBreak"]

    # Plot each pitch type as cluster
    pitch_types = sorted(mov["TaggedPitchType"].dropna().unique())
    for pt in pitch_types:
        sub = mov[mov["TaggedPitchType"] == pt]
        if len(sub) < 3:
            continue
        color = PITCH_COLORS.get(pt, "#aaa")
        fig.add_trace(go.Scatter(
            x=sub["HB_plot"], y=sub["InducedVertBreak"], mode="markers",
            marker=dict(size=7, color=color, opacity=0.7, line=dict(width=0.5, color="white")),
            name=pt,
            hovertemplate=f"{pt}<br>HB: %{{customdata:.1f}}\"<br>IVB: %{{y:.1f}}\"<extra></extra>",
            customdata=sub["HorzBreak"],
        ))

    # Axis labels — Savant convention: 1B ◄ MOVES TOWARD ► 3B
    fig.add_annotation(x=0, y=27, text="MORE RISE", showarrow=False,
                       font=dict(size=9, color="#666", family="Inter"), yshift=5)
    fig.add_annotation(x=0, y=-27, text="MORE DROP", showarrow=False,
                       font=dict(size=9, color="#666", family="Inter"), yshift=-5)
    fig.add_annotation(x=-27, y=0, text="1B SIDE", showarrow=False,
                       font=dict(size=9, color="#666", family="Inter"), xshift=-10)
    fig.add_annotation(x=27, y=0, text="3B SIDE", showarrow=False,
                       font=dict(size=9, color="#666", family="Inter"), xshift=10)
    # Direction arrow
    fig.add_annotation(x=0, y=29, text="1B ◄  MOVES TOWARD  ► 3B", showarrow=False,
                       font=dict(size=8, color="#999", family="Inter"), yshift=12)

    max_r = 28
    fig.update_layout(
        xaxis=dict(range=[-max_r, max_r], showgrid=False, zeroline=False,
                   showticklabels=False, title="", fixedrange=True, scaleanchor="y"),
        yaxis=dict(range=[-max_r, max_r], showgrid=False, zeroline=False,
                   showticklabels=False, title="", fixedrange=True),
        height=height, plot_bgcolor="#f0f4f8", paper_bgcolor="white",
        font=dict(color="#1a1a2e", family="Inter, Arial, sans-serif"),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                    font=dict(size=11, color="#1a1a2e")),
    )
    return fig


def make_pitch_location_heatmap(pitch_data, title, color, height=380):
    """Create a Savant-style location heatmap with blue-white-red density."""
    loc = pitch_data.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    if len(loc) < 3:
        return None

    fig = go.Figure()

    # Density contour — blue (sparse) → white → red (dense), matching Savant style
    fig.add_trace(go.Histogram2dContour(
        x=loc["PlateLocSide"], y=loc["PlateLocHeight"],
        colorscale=[
            (0.0, "rgba(255,255,255,0)"),
            (0.15, "rgba(173,203,227,0.6)"),
            (0.3, "rgba(120,170,210,0.7)"),
            (0.45, "rgba(180,200,220,0.6)"),
            (0.55, "rgba(230,220,220,0.6)"),
            (0.7, "rgba(230,180,180,0.7)"),
            (0.85, "rgba(220,120,120,0.8)"),
            (1.0, "rgba(200,60,60,0.9)"),
        ],
        showscale=False, ncontours=12,
        contours=dict(showlines=True, coloring="fill"),
        line=dict(width=0.5, color="rgba(150,150,150,0.3)"),
        hoverinfo="skip",
    ))

    # Strike zone outer box
    fig.add_shape(type="rect", x0=-ZONE_SIDE, x1=ZONE_SIDE, y0=ZONE_HEIGHT_BOT, y1=ZONE_HEIGHT_TOP,
                  line=dict(color="#333", width=2), fillcolor="rgba(0,0,0,0)")

    # Home plate
    plate_x = [-0.71, 0.71, 0.71, 0, -0.71, -0.71]
    plate_y = [0.15, 0.15, 0.0, -0.2, 0.0, 0.15]
    fig.add_trace(go.Scatter(x=plate_x, y=plate_y, mode="lines",
                             fill="toself", fillcolor="rgba(220,220,220,0.5)",
                             line=dict(color="#aaa", width=1.5), showlegend=False, hoverinfo="skip"))

    fig.update_layout(
        xaxis=dict(range=[-2.2, 2.2], showgrid=False, zeroline=False, showticklabels=False,
                   title="", fixedrange=True, scaleanchor="y"),
        yaxis=dict(range=[-0.5, 4.8], showgrid=False, zeroline=False, showticklabels=False,
                   title="", fixedrange=True),
        height=height, margin=dict(l=5, r=5, t=5, b=5),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="#1a1a2e", family="Inter, Arial, sans-serif"),
    )
    return fig


# ──────────────────────────────────────────────
# PITCHING OVERVIEW (used by page_pitching)
# ──────────────────────────────────────────────
def _pitching_overview(data, pitcher, season_filter, pdf, pdf_raw, pr, all_pitcher_stats):
    """Content from the original Pitcher Card, rendered inside the Overview tab."""
    all_stats = all_pitcher_stats  # alias for brevity
    total_pitches = len(pdf)

    # ── ROW 1: Percentile Rankings + Movement Profile ──
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        p_metrics = [
            ("FB Velo", pr["AvgFBVelo"], get_percentile(pr["AvgFBVelo"], all_stats["AvgFBVelo"]), ".1f", True),
            ("Avg EV Against", pr["AvgEVAgainst"], get_percentile(pr["AvgEVAgainst"], all_stats["AvgEVAgainst"]), ".1f", False),
            ("Chase %", pr["ChasePct"], get_percentile(pr["ChasePct"], all_stats["ChasePct"]), ".1f", True),
            ("Whiff %", pr["WhiffPct"], get_percentile(pr["WhiffPct"], all_stats["WhiffPct"]), ".1f", True),
            ("K %", pr["KPct"], get_percentile(pr["KPct"], all_stats["KPct"]), ".1f", True),
            ("BB %", pr["BBPct"], get_percentile(pr["BBPct"], all_stats["BBPct"]), ".1f", False),
            ("Barrel %", pr["BarrelPctAgainst"], get_percentile(pr["BarrelPctAgainst"], all_stats["BarrelPctAgainst"]), ".1f", False),
            ("Hard Hit %", pr["HardHitAgainst"], get_percentile(pr["HardHitAgainst"], all_stats["HardHitAgainst"]), ".1f", False),
            ("GB %", pr["GBPct"], get_percentile(pr["GBPct"], all_stats["GBPct"]), ".1f", True),
            ("Extension", pr["Extension"], get_percentile(pr["Extension"], all_stats["Extension"]), ".1f", True),
        ]
        render_savant_percentile_section(p_metrics, "Percentile Rankings")
        st.caption(f"vs. {len(all_stats)} pitchers in database (min 100 pitches)")

    with col2:
        section_header("Movement Profile (Induced Break)")
        fig = make_movement_profile(pdf, height=500)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="pitcher_movement")
        else:
            st.info("No movement data.")

    # ── ROW 2: Pitch Arsenal Table ──
    st.markdown("---")
    section_header("Pitch Arsenal")
    arsenal_rows = []
    main_pitches = sorted(pdf["TaggedPitchType"].dropna().unique())
    for pt in main_pitches:
        sub = pdf[pdf["TaggedPitchType"] == pt]
        n = len(sub)
        sub_swings = sub[sub["PitchCall"].isin(SWING_CALLS)]
        sub_whiffs = sub[sub["PitchCall"] == "StrikeSwinging"]
        sub_ip = sub[sub["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
        row = {
            "Pitch": pt,
            "#": n,
            "Use%": round(n / total_pitches * 100, 1),
            "Velo": round(sub["RelSpeed"].mean(), 1) if sub["RelSpeed"].notna().any() else None,
            "Max": round(sub["RelSpeed"].max(), 1) if sub["RelSpeed"].notna().any() else None,
            "Spin": int(round(sub["SpinRate"].mean())) if sub["SpinRate"].notna().any() else None,
            "IVB": round(sub["InducedVertBreak"].mean(), 1) if sub["InducedVertBreak"].notna().any() else None,
            "HB": round(sub["HorzBreak"].mean(), 1) if sub["HorzBreak"].notna().any() else None,
            "Ext.": round(sub["Extension"].mean(), 1) if sub["Extension"].notna().any() else None,
            "Whiff%": round(len(sub_whiffs) / max(len(sub_swings), 1) * 100, 1),
            "EV Ag": round(sub_ip["ExitSpeed"].mean(), 1) if len(sub_ip) > 0 else None,
        }
        if "Tilt" in sub.columns and sub["Tilt"].notna().any():
            row["Tilt"] = safe_mode(sub["Tilt"], None)
        if "ZoneSpeed" in sub.columns and sub["ZoneSpeed"].notna().any():
            row["Zone Velo"] = round(sub["ZoneSpeed"].mean(), 1)
        if "VertApprAngle" in sub.columns and sub["VertApprAngle"].notna().any():
            row["VAA"] = round(sub["VertApprAngle"].mean(), 1)
        arsenal_rows.append(row)
    if arsenal_rows:
        st.dataframe(pd.DataFrame(arsenal_rows), use_container_width=True, hide_index=True)

    # Arsenal summary text
    arsenal_summary = ", ".join(
        f"{r['Pitch']} ({r['Use%']}%)" for r in arsenal_rows
    )
    st.markdown(
        f'<p style="font-size:13px;color:#555;margin-top:4px;">'
        f'{display_name(pitcher)} relies on {len(main_pitches)} pitches: {arsenal_summary}</p>',
        unsafe_allow_html=True,
    )

    # ── ROW 3: Per-Pitch Location Heatmaps (Savant style) ──
    st.markdown("---")
    section_header("Pitch Locations by Type")

    # Arrange pitch heatmaps in a grid (up to 4 per row)
    n_pitches = len(main_pitches)
    cols_per_row = min(n_pitches, 4)
    for row_start in range(0, n_pitches, cols_per_row):
        row_pitches = main_pitches[row_start:row_start + cols_per_row]
        cols = st.columns(len(row_pitches), gap="small")
        for i, pt in enumerate(row_pitches):
            with cols[i]:
                sub = pdf[pdf["TaggedPitchType"] == pt]
                n = len(sub)
                color = PITCH_COLORS.get(pt, "#aaa")
                st.markdown(
                    f'<div style="font-size:14px;font-weight:700;color:{color} !important;">{pt}</div>'
                    f'<div style="font-size:11px;color:#888;">{n} Pitches ({n / total_pitches * 100:.1f}%)</div>',
                    unsafe_allow_html=True,
                )
                fig = make_pitch_location_heatmap(sub, pt, color, height=320)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"loc_{pt}_{row_start}")

    # ── ROW 4: Velocity Distribution + Usage Chart ──
    st.markdown("---")
    col3, col4 = st.columns([1, 1], gap="medium")

    with col3:
        section_header("Velocity Distribution")
        velo_data = pdf.dropna(subset=["RelSpeed"])
        if not velo_data.empty:
            fig = go.Figure()
            for pt in main_pitches:
                sub = velo_data[velo_data["TaggedPitchType"] == pt]
                if len(sub) < 5:
                    continue
                fig.add_trace(go.Violin(
                    y=sub["RelSpeed"], name=pt,
                    line_color=PITCH_COLORS.get(pt, "#aaa"),
                    fillcolor=PITCH_COLORS.get(pt, "#aaa"),
                    opacity=0.6, meanline_visible=True,
                    box_visible=True, box_fillcolor="white",
                ))
            fig.update_layout(
                yaxis_title="Velocity (mph)", showlegend=False,
                height=450, **CHART_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True, key="pitcher_velo_dist")

    with col4:
        section_header("Pitch Usage %")
        if arsenal_rows:
            usage_df = pd.DataFrame(arsenal_rows).sort_values("Use%", ascending=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=usage_df["Pitch"], x=usage_df["Use%"],
                orientation="h",
                marker_color=[PITCH_COLORS.get(p, "#aaa") for p in usage_df["Pitch"]],
                text=[f'{v}%' for v in usage_df["Use%"]],
                textposition="outside",
                textfont=dict(color="#1a1a2e", size=12, family="Inter"),
            ))
            fig.update_layout(
                xaxis_title="Usage %", yaxis_title="",
                height=450, showlegend=False,
                **CHART_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True, key="pitcher_usage")

    # ── ROW 5: Plate Discipline + Whiff by Type ──
    st.markdown("---")
    col5, col6 = st.columns([1, 1], gap="medium")

    with col5:
        section_header("Plate Discipline")
        disc_df = pd.DataFrame([{
            "Pitches": int(pr["Pitches"]),
            "Zone%": round(pr["ZonePct"], 1) if not pd.isna(pr["ZonePct"]) else None,
            "Chase%": round(pr["ChasePct"], 1) if not pd.isna(pr["ChasePct"]) else None,
            "Whiff%": round(pr["WhiffPct"], 1),
            "Z-Contact%": round(pr["ZoneContactPct"], 1) if not pd.isna(pr["ZoneContactPct"]) else None,
            "Swing%": round(pr["SwingPct"], 1),
            "K%": round(pr["KPct"], 1),
            "BB%": round(pr["BBPct"], 1),
        }])
        st.dataframe(disc_df, use_container_width=True, hide_index=True)

    with col6:
        section_header("Whiff% by Pitch Type")
        if arsenal_rows:
            whiff_df = pd.DataFrame(arsenal_rows)[["Pitch", "Whiff%"]].sort_values("Whiff%", ascending=False)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=whiff_df["Pitch"], y=whiff_df["Whiff%"],
                marker_color=[PITCH_COLORS.get(p, "#aaa") for p in whiff_df["Pitch"]],
                text=[f'{v}%' for v in whiff_df["Whiff%"]],
                textposition="outside",
                textfont=dict(color="#1a1a2e", size=12, family="Inter"),
            ))
            fig.update_layout(
                yaxis_title="Whiff %", height=350, showlegend=False, **CHART_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True, key="pitcher_whiff_by_type")

    # ── ROW 6: Velocity + Spin Over Time (full width, one per row) ──
    st.markdown("---")
    section_header("Velocity Over Time")
    fb_data = pdf.dropna(subset=["RelSpeed", "Date"])
    if not fb_data.empty:
        daily = fb_data.groupby(["Date", "TaggedPitchType"])["RelSpeed"].mean().reset_index()
        fig = px.line(daily, x="Date", y="RelSpeed", color="TaggedPitchType",
                      color_discrete_map=PITCH_COLORS, markers=True,
                      labels={"RelSpeed": "Velo (mph)", "Date": ""})
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                        font=dict(size=11, color="#1a1a2e")),
            **CHART_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True, key="pitcher_velo_time")

    section_header("Spin Rate Over Time")
    spin_data = pdf.dropna(subset=["SpinRate", "Date"])
    if not spin_data.empty:
        daily_spin = spin_data.groupby(["Date", "TaggedPitchType"])["SpinRate"].mean().reset_index()
        fig = px.line(daily_spin, x="Date", y="SpinRate", color="TaggedPitchType",
                      color_discrete_map=PITCH_COLORS, markers=True,
                      labels={"SpinRate": "Spin (rpm)", "Date": ""})
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                        font=dict(size=11, color="#1a1a2e")),
            **CHART_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True, key="pitcher_spin_time")

    # ── ROW 7: Release Point Consistency ──
    st.markdown("---")
    section_header("Release Point Consistency")
    rp_data = pdf.dropna(subset=["RelHeight", "RelSide"])
    if len(rp_data) >= 10:
        col_rp1, col_rp2 = st.columns([1, 1], gap="medium")
        with col_rp1:
            fig = go.Figure()
            for pt in main_pitches:
                sub = rp_data[rp_data["TaggedPitchType"] == pt]
                if len(sub) < 3:
                    continue
                fig.add_trace(go.Scatter(
                    x=sub["RelSide"], y=sub["RelHeight"], mode="markers",
                    marker=dict(size=5, color=PITCH_COLORS.get(pt, "#aaa"), opacity=0.6,
                                line=dict(width=0.3, color="white")),
                    name=pt,
                    hovertemplate=f"{pt}<br>Side: %{{x:.2f}} ft<br>Height: %{{y:.2f}} ft<extra></extra>",
                ))
                # Add mean crosshair
                fig.add_trace(go.Scatter(
                    x=[sub["RelSide"].mean()], y=[sub["RelHeight"].mean()], mode="markers",
                    marker=dict(size=14, color=PITCH_COLORS.get(pt, "#aaa"), symbol="x-thin",
                                line=dict(width=3, color=PITCH_COLORS.get(pt, "#aaa"))),
                    showlegend=False, hoverinfo="skip",
                ))
            fig.update_layout(
                xaxis_title="Release Side (ft)", yaxis_title="Release Height (ft)",
                height=400,
                xaxis=dict(scaleanchor="y"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                            font=dict(size=10, color="#1a1a2e")),
                **CHART_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True, key="pitcher_rp")

        with col_rp2:
            # Release point consistency stats
            rp_rows = []
            for pt in main_pitches:
                sub = rp_data[rp_data["TaggedPitchType"] == pt]
                if len(sub) < 5:
                    continue
                rp_rows.append({
                    "Pitch": pt,
                    "Avg Height": round(sub["RelHeight"].mean(), 2),
                    "Std Height": round(sub["RelHeight"].std(), 3),
                    "Avg Side": round(sub["RelSide"].mean(), 2),
                    "Std Side": round(sub["RelSide"].std(), 3),
                })
            if rp_rows:
                st.dataframe(pd.DataFrame(rp_rows), use_container_width=True, hide_index=True)
                # Insight
                best_rp = min(rp_rows, key=lambda r: r["Std Height"] + r["Std Side"])
                worst_rp = max(rp_rows, key=lambda r: r["Std Height"] + r["Std Side"])
                st.markdown(
                    f'<p style="font-size:12px;color:#555;">'
                    f'Most consistent release: <b style="color:{PITCH_COLORS.get(best_rp["Pitch"], "#333")}">'
                    f'{best_rp["Pitch"]}</b> (std: {best_rp["Std Height"]:.3f}H / {best_rp["Std Side"]:.3f}S). '
                    f'Least consistent: <b style="color:{PITCH_COLORS.get(worst_rp["Pitch"], "#333")}">'
                    f'{worst_rp["Pitch"]}</b> (std: {worst_rp["Std Height"]:.3f}H / {worst_rp["Std Side"]:.3f}S).</p>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("Not enough release point data.")

    # ── ROW 8: Approach Angle (VAA) Analysis ──
    st.markdown("---")
    section_header("Vertical Approach Angle (VAA)")
    vaa_data = pdf.dropna(subset=["VertApprAngle"])
    if len(vaa_data) >= 10:
        col_vaa1, col_vaa2 = st.columns([1, 1], gap="medium")
        with col_vaa1:
            fig = go.Figure()
            for pt in main_pitches:
                sub = vaa_data[vaa_data["TaggedPitchType"] == pt]
                if len(sub) < 5:
                    continue
                fig.add_trace(go.Violin(
                    y=sub["VertApprAngle"], name=pt,
                    line_color=PITCH_COLORS.get(pt, "#aaa"),
                    fillcolor=PITCH_COLORS.get(pt, "#aaa"),
                    opacity=0.6, meanline_visible=True,
                    box_visible=True, box_fillcolor="white",
                ))
            fig.update_layout(
                yaxis_title="VAA (degrees)", showlegend=False,
                height=380, **CHART_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True, key="pitcher_vaa")

        with col_vaa2:
            # VAA percentile context
            vaa_rows = []
            all_pitchers_data = data.dropna(subset=["VertApprAngle"])
            for pt in main_pitches:
                sub = vaa_data[vaa_data["TaggedPitchType"] == pt]
                if len(sub) < 5:
                    continue
                p_vaa = sub["VertApprAngle"].mean()
                # Get all pitchers' avg VAA for this pitch type
                all_pt = all_pitchers_data[all_pitchers_data["TaggedPitchType"] == pt]
                pitcher_avgs = all_pt.groupby("Pitcher")["VertApprAngle"].mean()
                pitcher_avgs = pitcher_avgs[all_pt.groupby("Pitcher").size() >= 20]
                pct = get_percentile(p_vaa, pitcher_avgs) if len(pitcher_avgs) > 0 else np.nan
                # For fastballs, flatter (less negative) VAA is better; for breaking, steeper is better
                is_fb = pt in ["Fastball", "Sinker", "Cutter"]
                vaa_rows.append((pt, p_vaa, pct, ".1f", is_fb))
            if vaa_rows:
                render_savant_percentile_section(vaa_rows, "VAA Percentile by Pitch")
                st.caption("Fastballs: flatter VAA is elite. Breaking: steeper is better.")
    else:
        st.info("No VAA data available.")

    # ── ROW 9: Platoon Splits (vs LHH / RHH) ──
    st.markdown("---")
    section_header("Platoon Splits (vs LHH / RHH)")
    if "BatterSide" in pdf.columns:
        platoon_metrics = []
        for side, label in [("Right", "vs RHH"), ("Left", "vs LHH")]:
            p_sub = pdf_raw[pdf_raw["BatterSide"] == side]
            if len(p_sub) < 10:
                continue
            p_sw = p_sub[p_sub["PitchCall"].isin(SWING_CALLS)]
            p_wh = p_sub[p_sub["PitchCall"] == "StrikeSwinging"]
            p_ip = p_sub[p_sub["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            p_whiff = len(p_wh) / max(len(p_sw), 1) * 100 if len(p_sw) > 0 else np.nan
            p_ev = p_ip["ExitSpeed"].mean() if len(p_ip) > 0 else np.nan
            # Get all pitchers' whiff/ev vs this side (via DuckDB)
            all_df = query_population(f"""
                SELECT
                    Pitcher,
                    CASE WHEN SUM(CASE WHEN PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END) > 0
                        THEN SUM(CASE WHEN PitchCall='StrikeSwinging' THEN 1 ELSE 0 END) * 100.0
                             / SUM(CASE WHEN PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END)
                        ELSE NULL END AS whiff,
                    AVG(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN ExitSpeed END) AS ev
                FROM trackman
                WHERE BatterSide = '{side}'
                GROUP BY Pitcher
                HAVING COUNT(*) >= 20
            """)
            whiff_pct = get_percentile(p_whiff, all_df["whiff"]) if not all_df.empty and not pd.isna(p_whiff) else np.nan
            ev_pct = get_percentile(p_ev, all_df["ev"]) if not all_df.empty and not pd.isna(p_ev) else np.nan
            platoon_metrics.append((f"{label} Whiff%", p_whiff, whiff_pct, ".1f", True))
            if not pd.isna(ev_pct):
                platoon_metrics.append((f"{label} EV Against", p_ev, ev_pct, ".1f", False))

        if platoon_metrics:
            render_savant_percentile_section(platoon_metrics, None)
            st.caption("Percentile vs. all pitchers in DB (min 100 pitches vs that side)")

            # Usage breakdown by side
            col_p1, col_p2 = st.columns(2, gap="medium")
            for i, (side, label) in enumerate([("Right", "vs RHH"), ("Left", "vs LHH")]):
                p_sub = pdf_raw[pdf_raw["BatterSide"] == side]
                if len(p_sub) < 10:
                    continue
                usage = p_sub["TaggedPitchType"].value_counts(normalize=True).mul(100).round(1)
                with [col_p1, col_p2][i]:
                    st.markdown(f"**Pitch Usage {label}** ({len(p_sub)} pitches)")
                    for pt, pct_val in usage.items():
                        color = PITCH_COLORS.get(pt, "#aaa")
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0;">'
                            f'<div style="width:12px;height:12px;border-radius:50%;background:{color};"></div>'
                            f'<div style="font-size:12px;color:#1a1a2e;">{pt}: <b>{pct_val:.1f}%</b></div>'
                            f'</div>', unsafe_allow_html=True,
                        )
    else:
        st.caption("No batter side data available.")

    # ── PitcherSet Splits (Stretch vs Windup) ──
    if "PitcherSet" in pdf_raw.columns and pdf_raw["PitcherSet"].notna().any():
        sets = pdf_raw["PitcherSet"].dropna().unique()
        if len(sets) > 1:
            st.markdown("---")
            section_header("Set Position Splits (Windup vs Stretch)")
            set_metrics = []
            for s_val in sorted(sets):
                sub = pdf_raw[pdf_raw["PitcherSet"] == s_val]
                if len(sub) < 10:
                    continue
                sw = sub[sub["PitchCall"].isin(SWING_CALLS)]
                wh = sub[sub["PitchCall"] == "StrikeSwinging"]
                fb = sub[sub["TaggedPitchType"].isin(["Fastball", "Sinker", "Cutter"])]
                fb_velo = fb["RelSpeed"].mean() if len(fb) > 0 and fb["RelSpeed"].notna().any() else np.nan
                whiff = len(wh) / max(len(sw), 1) * 100 if len(sw) > 0 else np.nan
                # Context vs all pitchers in that set (via DuckDB)
                all_df = query_population(f"""
                    SELECT
                        Pitcher,
                        CASE WHEN SUM(CASE WHEN PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END) > 0
                            THEN SUM(CASE WHEN PitchCall='StrikeSwinging' THEN 1 ELSE 0 END) * 100.0
                                 / SUM(CASE WHEN PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END)
                            ELSE NULL END AS whiff,
                        AVG(CASE WHEN TaggedPitchType IN ('Fastball','Sinker','Cutter') AND RelSpeed IS NOT NULL
                            THEN RelSpeed END) AS velo
                    FROM trackman
                    WHERE PitcherSet = '{s_val}'
                    GROUP BY Pitcher
                    HAVING COUNT(*) >= 20
                """)
                w_pct = get_percentile(whiff, all_df["whiff"]) if not all_df.empty and not pd.isna(whiff) else np.nan
                v_pct = get_percentile(fb_velo, all_df["velo"]) if not all_df.empty and not pd.isna(fb_velo) else np.nan
                set_metrics.append((f"{s_val} Whiff%", whiff, w_pct, ".1f", True))
                if not pd.isna(v_pct):
                    set_metrics.append((f"{s_val} FB Velo", fb_velo, v_pct, ".1f", True))
            if set_metrics:
                render_savant_percentile_section(set_metrics, None)

    # ── ROW 10: Pitch Sequencing ──
    st.markdown("---")
    section_header("Pitch Sequencing Patterns")
    if "PitchofPA" in pdf_raw.columns and "TaggedPitchType" in pdf_raw.columns:
        seq_data = pdf_raw.dropna(subset=["PitchofPA", "TaggedPitchType"]).copy()
        # First pitch tendencies
        first_pitches = seq_data[seq_data["PitchofPA"] == 1]
        if len(first_pitches) >= 5:
            fp_usage = first_pitches["TaggedPitchType"].value_counts(normalize=True).mul(100).round(1)
            col_s1, col_s2 = st.columns([1, 1], gap="medium")
            with col_s1:
                st.markdown("**First Pitch Tendencies**")
                fig = go.Figure()
                fig.add_trace(go.Pie(
                    labels=fp_usage.index, values=fp_usage.values,
                    marker_colors=[PITCH_COLORS.get(p, "#aaa") for p in fp_usage.index],
                    hole=0.4, textinfo="label+percent",
                    textfont=dict(size=11, color="#1a1a2e"),
                ))
                fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                                  plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
                st.plotly_chart(fig, use_container_width=True, key="pitcher_fp_pie")

                # First pitch strike%
                fp_strikes = first_pitches[first_pitches["PitchCall"].isin(
                    ["StrikeCalled", "StrikeSwinging", "FoulBall", "FoulBallNotFieldable",
                     "FoulBallFieldable", "InPlay"])]
                fp_strike_pct = len(fp_strikes) / max(len(first_pitches), 1) * 100
                # Context: all pitchers' first pitch strike% (via DuckDB)
                _fp_df = query_population(f"""
                    SELECT Pitcher,
                        SUM(CASE WHEN PitchCall IN ('StrikeCalled','StrikeSwinging','FoulBall',
                            'FoulBallNotFieldable','FoulBallFieldable','InPlay') THEN 1 ELSE 0 END) * 100.0
                            / GREATEST(COUNT(*), 1) AS fp_strike
                    FROM trackman
                    WHERE PitchofPA = 1
                    GROUP BY Pitcher
                    HAVING COUNT(*) >= 10
                """)
                fp_pct = get_percentile(fp_strike_pct, _fp_df["fp_strike"]) if not _fp_df.empty else 50
                render_savant_percentile_section(
                    [("1st Pitch Strike%", fp_strike_pct, fp_pct, ".1f", True)], None,
                )

            with col_s2:
                # Two-strike approach
                two_strike = seq_data[(seq_data["Strikes"] == 2)]
                if len(two_strike) >= 10:
                    st.markdown("**Two-Strike Approach**")
                    ts_usage = two_strike["TaggedPitchType"].value_counts(normalize=True).mul(100).round(1)
                    fig = go.Figure()
                    fig.add_trace(go.Pie(
                        labels=ts_usage.index, values=ts_usage.values,
                        marker_colors=[PITCH_COLORS.get(p, "#aaa") for p in ts_usage.index],
                        hole=0.4, textinfo="label+percent",
                        textfont=dict(size=11, color="#1a1a2e"),
                    ))
                    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                                      plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True, key="pitcher_ts_pie")

                    ts_whiffs = two_strike[two_strike["PitchCall"] == "StrikeSwinging"]
                    ts_sw = two_strike[two_strike["PitchCall"].isin(SWING_CALLS)]
                    ts_whiff_pct = len(ts_whiffs) / max(len(ts_sw), 1) * 100
                    st.markdown(
                        f'<p style="font-size:12px;color:#555;">Two-strike whiff rate: '
                        f'<b>{ts_whiff_pct:.1f}%</b> on {len(ts_sw)} swings</p>',
                        unsafe_allow_html=True,
                    )


# ──────────────────────────────────────────────
# PITCHER CARD — Consolidated Summary Tab
# ──────────────────────────────────────────────
def _pitcher_card_content(data, pitcher, season_filter, pdf, stuff_df, pr, all_pitcher_stats):
    """Render a single-page Pitcher Card with arsenal, locations, tunnels,
    sequences, and platoon splits."""
    all_stats = all_pitcher_stats
    total_pitches = len(pdf)

    # ── Percentile Rankings + Arsenal Table side by side ──
    pc_col1, pc_col2 = st.columns([1, 1], gap="medium")
    with pc_col1:
        p_metrics = [
            ("FB Velo", pr["AvgFBVelo"], get_percentile(pr["AvgFBVelo"], all_stats["AvgFBVelo"]), ".1f", True),
            ("Avg EV Against", pr["AvgEVAgainst"], get_percentile(pr["AvgEVAgainst"], all_stats["AvgEVAgainst"]), ".1f", False),
            ("Chase %", pr["ChasePct"], get_percentile(pr["ChasePct"], all_stats["ChasePct"]), ".1f", True),
            ("Whiff %", pr["WhiffPct"], get_percentile(pr["WhiffPct"], all_stats["WhiffPct"]), ".1f", True),
            ("K %", pr["KPct"], get_percentile(pr["KPct"], all_stats["KPct"]), ".1f", True),
            ("BB %", pr["BBPct"], get_percentile(pr["BBPct"], all_stats["BBPct"]), ".1f", False),
            ("Barrel %", pr["BarrelPctAgainst"], get_percentile(pr["BarrelPctAgainst"], all_stats["BarrelPctAgainst"]), ".1f", False),
            ("Hard Hit %", pr["HardHitAgainst"], get_percentile(pr["HardHitAgainst"], all_stats["HardHitAgainst"]), ".1f", False),
            ("GB %", pr["GBPct"], get_percentile(pr["GBPct"], all_stats["GBPct"]), ".1f", True),
            ("Extension", pr["Extension"], get_percentile(pr["Extension"], all_stats["Extension"]), ".1f", True),
        ]
        render_savant_percentile_section(p_metrics, "Percentile Rankings")
        st.caption(f"vs. {len(all_stats)} pitchers in database (min 100 pitches)")

    with pc_col2:
        section_header("Movement Profile (Induced Break)")
        fig_mov = make_movement_profile(pdf, height=420)
        if fig_mov:
            st.plotly_chart(fig_mov, use_container_width=True, key="pc_movement")
        else:
            st.info("No movement data.")

        # Usage & Velo summary underneath
        main_pitches_sorted = pdf["TaggedPitchType"].value_counts().index.tolist()
        uv_cols = st.columns(min(len(main_pitches_sorted), 5))
        for idx, pt in enumerate(main_pitches_sorted[:5]):
            with uv_cols[idx]:
                sub = pdf[pdf["TaggedPitchType"] == pt]
                usage = len(sub) / total_pitches * 100
                velo = sub["RelSpeed"].mean()
                pc = PITCH_COLORS.get(pt, "#888")
                velo_str = f"{velo:.1f}" if not pd.isna(velo) else "-"
                st.markdown(
                    f'<div style="text-align:center;padding:6px 4px;border-radius:6px;'
                    f'border-top:3px solid {pc};background:#f9f9f9;">'
                    f'<div style="font-weight:bold;font-size:13px;color:{pc};">{pt}</div>'
                    f'<div style="font-size:12px;color:#555;">{velo_str} mph</div>'
                    f'<div style="font-size:11px;color:#888;">{usage:.0f}%</div>'
                    f'</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Section A: Stuff+ & Command+ Grades ──
    section_header("Stuff+ & Command+ Grades")
    has_stuff = stuff_df is not None and "StuffPlus" in stuff_df.columns and not stuff_df.empty
    if has_stuff:
        arsenal_agg = stuff_df.groupby("TaggedPitchType").agg(
            stuff_mean=("StuffPlus", "mean"),
            velo_mean=("RelSpeed", "mean"),
            count=("StuffPlus", "count"),
        )
        arsenal_agg["Usage%"] = (arsenal_agg["count"] / arsenal_agg["count"].sum() * 100).round(1)
        arsenal_agg = arsenal_agg.sort_values("Usage%", ascending=False)
    else:
        # Fallback: derive from pdf directly
        pt_counts = pdf["TaggedPitchType"].value_counts()
        arsenal_agg = pd.DataFrame({
            "stuff_mean": np.nan,
            "velo_mean": pdf.groupby("TaggedPitchType")["RelSpeed"].mean(),
            "count": pt_counts,
            "Usage%": (pt_counts / pt_counts.sum() * 100).round(1),
        }).sort_values("Usage%", ascending=False)

    cmd_df = _compute_command_plus(pdf, data)
    cmd_map = {}
    if not cmd_df.empty:
        cmd_map = dict(zip(cmd_df["Pitch"], cmd_df["Command+"]))

    pitch_types = arsenal_agg.index.tolist()
    n_pitches = len(pitch_types)

    # Stuff+ percentile bars
    if has_stuff:
        all_stuff = _compute_stuff_plus(data)
        if "StuffPlus" in all_stuff.columns:
            stuff_metrics = []
            for pt in pitch_types:
                my_val = stuff_df[stuff_df["TaggedPitchType"] == pt]["StuffPlus"].mean()
                all_pt_vals = all_stuff[all_stuff["TaggedPitchType"] == pt]["StuffPlus"]
                if len(all_pt_vals) > 5:
                    pctl = percentileofscore(all_pt_vals.dropna(), my_val, kind="rank")
                    stuff_metrics.append((pt, my_val, pctl, ".0f", True))
            if stuff_metrics:
                render_savant_percentile_section(stuff_metrics,
                                                 title="Stuff+ Percentile Rankings")

    # Command+ percentile bars
    if not cmd_df.empty:
        cmd_metrics = []
        for _, row in cmd_df.iterrows():
            cmd_val = row["Command+"]
            pctl_mapped = min(max((cmd_val - 80) * 2.5, 0), 100)
            cmd_metrics.append((row["Pitch"], cmd_val, pctl_mapped, ".0f", True))
        if cmd_metrics:
            render_savant_percentile_section(cmd_metrics,
                                             title="Command+ Percentile Rankings")

    # ── Section B: Best Pitch Locations (whiffs, called strikes, weak contact) ──
    st.markdown("")
    section_header("Best Pitch Locations")
    sel_pt = st.selectbox("Select Pitch", pitch_types, key="pc_loc_pitch")
    pt_data = pdf[pdf["TaggedPitchType"] == sel_pt].dropna(subset=["PlateLocSide", "PlateLocHeight"])

    whiff_data = pt_data[pt_data["PitchCall"] == "StrikeSwinging"]
    cs_data = pt_data[pt_data["PitchCall"] == "StrikeCalled"]
    weak_data = pt_data[(pt_data["PitchCall"] == "InPlay") & (pt_data["ExitSpeed"] < 85)].dropna(subset=["ExitSpeed"])

    loc_defs = [
        ("Whiff Locations", whiff_data, "YlOrRd", f"{len(whiff_data)} whiffs"),
        ("Called Strike Locations", cs_data, "Blues", f"{len(cs_data)} called strikes"),
        ("Weak Contact Locations", weak_data, "Greens", f"{len(weak_data)} weak contacts (EV < 85)"),
    ]
    loc_cols = st.columns(3)
    for idx, (title, ldata, cscale, caption_txt) in enumerate(loc_defs):
        with loc_cols[idx]:
            section_header(title)
            if len(ldata) >= 3:
                fig = go.Figure()
                fig.add_trace(go.Histogram2d(
                    x=ldata["PlateLocSide"].values,
                    y=ldata["PlateLocHeight"].values,
                    nbinsx=8, nbinsy=8,
                    colorscale=cscale, showscale=False,
                ))
                add_strike_zone(fig)
                fig.update_layout(
                    height=280,
                    margin=dict(l=25, r=10, t=10, b=25),
                    xaxis=dict(range=[-2.5, 2.5], scaleanchor="y"),
                    yaxis=dict(range=[0, 5.5]),
                    plot_bgcolor="white", paper_bgcolor="white",
                )
                st.plotly_chart(fig, use_container_width=True, key=f"pc_loc_{sel_pt}_{idx}")
            else:
                st.caption("Not enough data")
            st.caption(caption_txt)

    # ── Section C: Tunnel Pairs ──
    if n_pitches >= 2:
        st.markdown("")
        section_header("Tunnel Pairs")
        tunnel_pop = build_tunnel_population_pop()
        tunnel_df = _compute_tunnel_score(pdf, tunnel_pop=tunnel_pop)
        # Filter to pitches with >= 10% usage
        usage_ok = set(arsenal_agg[arsenal_agg["Usage%"] >= 10].index)
        if not tunnel_df.empty:
            tunnel_df = tunnel_df[
                tunnel_df["Pitch A"].isin(usage_ok) & tunnel_df["Pitch B"].isin(usage_ok)
            ].reset_index(drop=True)
        if tunnel_df.empty:
            st.info("Need 2+ pitch types (≥10% usage) to compute tunnels.")
        else:
            grade_colors = {"A": "#22c55e", "B": "#3b82f6", "C": "#f59e0b",
                            "D": "#f97316", "F": "#ef4444"}
            # Show all tunnel pairs in a compact table
            for _, row in tunnel_df.iterrows():
                gc = grade_colors.get(row["Grade"], "#888")
                commit_str = f'{row["Commit Sep (in)"]:.1f}' if "Commit Sep (in)" in row and not pd.isna(row["Commit Sep (in)"]) else "-"
                st.markdown(
                    f'<div style="padding:10px 14px;border-radius:8px;'
                    f'border-left:4px solid {gc};background:{gc}10;margin:4px 0;">'
                    f'<span style="font-size:22px;font-weight:bold;color:{gc};">'
                    f'{row["Grade"]}</span>'
                    f'<span style="font-size:14px;font-weight:600;margin-left:10px;">'
                    f'{row["Pitch A"]} + {row["Pitch B"]}</span>'
                    f'<span style="font-size:12px;color:#666;margin-left:10px;">Score: {row["Tunnel Score"]}'
                    f' &middot; Commit Sep: {commit_str} in</span>'
                    f'</div>', unsafe_allow_html=True)
    else:
        tunnel_df = pd.DataFrame()

    # ── Section D: Best Sequences ──
    if n_pitches >= 2:
        st.markdown("")
        section_header("Best Sequences")
        pair_df = _compute_pitch_pair_results(pdf, data, tunnel_df=tunnel_df if not tunnel_df.empty else None)
        # Build sorted_ps from pitcher's own stats
        ps_items = []
        total = len(pdf)
        for pt in pitch_types:
            pt_d = pdf[pdf["TaggedPitchType"] == pt]
            if len(pt_d) < 10:
                continue
            usage = len(pt_d) / total * 100
            sw = pt_d[pt_d["PitchCall"].isin(SWING_CALLS)]
            whiff_r = len(pt_d[pt_d["PitchCall"] == "StrikeSwinging"]) / max(len(sw), 1) * 100
            out_z = ~in_zone_mask(pt_d)
            chase_sw = pt_d[out_z & pt_d["PitchCall"].isin(SWING_CALLS)]
            chase_r = len(chase_sw) / max(out_z.sum(), 1) * 100
            velo = pt_d["RelSpeed"].mean()
            eff_velo_vals = pt_d["EffectiveVelo"].dropna() if "EffectiveVelo" in pt_d.columns else pd.Series(dtype=float)
            eff_v = eff_velo_vals.mean() if len(eff_velo_vals) > 0 else np.nan
            ev_against = pt_d["ExitSpeed"].dropna()
            avg_ev = ev_against.mean() if len(ev_against) >= 3 else np.nan
            stuff_avg = np.nan
            if has_stuff:
                st_vals = stuff_df[stuff_df["TaggedPitchType"] == pt]["StuffPlus"]
                if len(st_vals) > 0:
                    stuff_avg = st_vals.mean()
            # Composite score
            vals = []
            if not pd.isna(stuff_avg):
                s_norm = min(max((stuff_avg - 70) / 60 * 100, 0), 100)
                vals.append(("s", s_norm))
            if whiff_r > 0:
                w_norm = min(whiff_r / 50 * 100, 100)
                vals.append(("w", w_norm))
            if chase_r > 0:
                c_norm = min(chase_r / 40 * 100, 100)
                vals.append(("c", c_norm))
            if vals:
                weights = {"s": 0.5, "w": 0.3, "c": 0.2}
                total_w = sum(weights[k] for k, _ in vals)
                comp = sum(weights[k] * v for k, v in vals) / total_w if total_w > 0 else 50
            else:
                comp = 50
            ps_items.append((pt, {
                "count": len(pt_d), "usage": usage, "score": comp,
                "our_whiff": whiff_r, "our_chase": chase_r,
                "velo": velo, "eff_velo": eff_v,
            }))
        ps_items.sort(key=lambda x: x[1]["score"], reverse=True)
        top_seqs = _build_3pitch_sequences(ps_items, {}, tunnel_df, pair_df)
        if top_seqs:
            n_show = min(len(top_seqs), 3)
            seq_cols = st.columns(n_show)
            grade_colors = {"A": "#22c55e", "B": "#3b82f6", "C": "#f59e0b",
                            "D": "#f97316", "F": "#ef4444"}
            for idx, seq in enumerate(top_seqs[:n_show]):
                with seq_cols[idx]:
                    t12 = seq.get("t12", np.nan)
                    t23 = seq.get("t23", np.nan)
                    _, g12 = _lookup_tunnel(seq["p1"], seq["p2"], tunnel_df) if not pd.isna(t12) else (np.nan, "-")
                    _, g23 = _lookup_tunnel(seq["p2"], seq["p3"], tunnel_df) if not pd.isna(t23) else (np.nan, "-")
                    gc12 = grade_colors.get(g12, "#888")
                    gc23 = grade_colors.get(g23, "#888")
                    st.markdown(
                        f'<div style="text-align:center;padding:12px;border-radius:8px;'
                        f'border:1px solid #ddd;background:#f9f9f9;margin:2px;">'
                        f'<div style="font-size:16px;font-weight:bold;">{seq["seq"]}</div>'
                        f'<div style="font-size:20px;font-weight:bold;color:#1a1a2e;">'
                        f'Score: {seq["score"]:.0f}</div>'
                        f'<div style="font-size:12px;color:#666;margin-top:4px;">'
                        f'T<sub>12</sub>: <span style="color:{gc12};font-weight:bold;">{g12}</span>'
                        f' &middot; T<sub>23</sub>: '
                        f'<span style="color:{gc23};font-weight:bold;">{g23}</span></div>'
                        f'</div>', unsafe_allow_html=True)
        else:
            st.info("Not enough pitch variety to build sequences.")

        # Transition matrix mini-heatmap
        sort_cols_tm = [c for c in ["GameID", "Batter", "PAofInning", "PitchNo"] if c in pdf.columns]
        if len(sort_cols_tm) >= 2:
            pdf_tm = pdf.sort_values(sort_cols_tm).copy()
            pdf_tm["NextPitch"] = pdf_tm.groupby(["GameID", "Batter", "PAofInning"])["TaggedPitchType"].shift(-1)
            trans = pdf_tm.dropna(subset=["NextPitch"])
            if not trans.empty and len(trans["TaggedPitchType"].unique()) >= 2:
                st.caption("Pitch Transition Frequencies")
                matrix = pd.crosstab(trans["TaggedPitchType"], trans["NextPitch"], normalize="index") * 100
                fig_matrix = px.imshow(
                    matrix.round(1), text_auto=".0f", color_continuous_scale="RdBu_r",
                    labels=dict(x="Next Pitch", y="Current Pitch", color="%"),
                    aspect="auto",
                )
                fig_matrix.update_layout(height=280, plot_bgcolor="white", paper_bgcolor="white",
                                         font=dict(size=11, color="#1a1a2e", family="Inter, Arial, sans-serif"),
                                         margin=dict(l=45, r=10, t=10, b=40))
                st.plotly_chart(fig_matrix, use_container_width=True, key="pc_trans_mini")

    # ── Section E: Platoon Splits ──
    st.markdown("")
    section_header("Platoon Splits")
    split_cols = st.columns(2)
    for side_idx, (side, label) in enumerate([("Right", "vs RHH"), ("Left", "vs LHH")]):
        with split_cols[side_idx]:
            st.markdown(f"**{label}**")
            side_pdf = pdf[pdf["BatterSide"] == side]
            if len(side_pdf) < 10:
                st.caption(f"Limited data vs {label} ({len(side_pdf)} pitches)")
                continue
            side_cmd = _compute_command_plus(side_pdf, data)
            side_cmd_map = dict(zip(side_cmd["Pitch"], side_cmd["Command+"])) if not side_cmd.empty else {}
            split_rows = []
            for pt in pitch_types:
                pt_d = side_pdf[side_pdf["TaggedPitchType"] == pt]
                if len(pt_d) < 10:
                    continue
                sw = pt_d[pt_d["PitchCall"].isin(SWING_CALLS)]
                whiff_pct = len(pt_d[pt_d["PitchCall"] == "StrikeSwinging"]) / max(len(sw), 1) * 100
                csw_pct = pt_d["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"]).mean() * 100
                cmd_plus = side_cmd_map.get(pt, np.nan)
                loc_vals = pt_d.dropna(subset=["PlateLocSide", "PlateLocHeight"])
                loc_spread = np.sqrt(loc_vals["PlateLocHeight"].std()**2 + loc_vals["PlateLocSide"].std()**2) if len(loc_vals) >= 5 else np.nan
                # Composite for ranking
                w_norm = min(whiff_pct / 50, 1)
                c_norm = min(csw_pct / 40, 1)
                cmd_norm = min(max((cmd_plus - 80) / 40, 0), 1) if not pd.isna(cmd_plus) else 0.5
                comp = w_norm * 0.4 + c_norm * 0.3 + cmd_norm * 0.3
                if not pd.isna(loc_spread):
                    if loc_spread < 0.8:
                        consist = "precise"
                    elif loc_spread < 1.1:
                        consist = "average"
                    else:
                        consist = "inconsistent"
                else:
                    consist = "-"
                split_rows.append({
                    "pitch": pt, "whiff": whiff_pct, "csw": csw_pct,
                    "cmd": cmd_plus, "consist": consist, "comp": comp,
                    "count": len(pt_d),
                })
            if not split_rows:
                st.caption("No pitch type with 10+ pitches")
                continue
            split_rows.sort(key=lambda x: x["comp"], reverse=True)
            for sr in split_rows[:3]:
                pc = PITCH_COLORS.get(sr["pitch"], "#888")
                cmd_str = f'{sr["cmd"]:.0f}' if not pd.isna(sr["cmd"]) else "-"
                st.markdown(
                    f'<div style="padding:6px 10px;border-left:4px solid {pc};'
                    f'background:#f9f9f9;border-radius:4px;margin-bottom:4px;font-size:13px;">'
                    f'<b style="color:{pc};">{sr["pitch"]}</b> '
                    f'<span style="color:#555;">({sr["count"]} pitches)</span><br>'
                    f'Whiff: <b>{sr["whiff"]:.0f}%</b> &middot; '
                    f'CSW: <b>{sr["csw"]:.0f}%</b> &middot; '
                    f'Cmd+: <b>{cmd_str}</b>'
                    f'</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# PITCH LAB: Recommendation Engine + Page
# ──────────────────────────────────────────────

_METRIC_LABELS = {
    "RelSpeed": "Velocity",
    "InducedVertBreak": "IVB",
    "HorzBreak": "Horizontal Break",
    "Extension": "Extension",
    "VertApprAngle": "VAA",
    "SpinRate": "Spin Rate",
}

_METRIC_UNITS = {
    "RelSpeed": "mph",
    "InducedVertBreak": "in",
    "HorzBreak": "in",
    "Extension": "ft",
    "VertApprAngle": "°",
    "SpinRate": "rpm",
}


def _compute_pitch_recommendations(pdf, data, tunnel_df):
    """For each pitch type, compute actionable improvement suggestions.

    Returns list of dicts: {pitch, metric, label, current, target, delta,
    direction, unit, tunnel_benefit}

    HorzBreak is handled via absolute values so LHP (negative HB) and RHP
    (positive HB) are compared on the same scale.  Displayed values are
    converted back to the pitcher's sign convention.
    """
    from scipy.stats import percentileofscore as _pctile

    # Stuff+ weights define which direction is "better" for each metric
    # For HorzBreak the weight sign describes *magnitude* direction after
    # taking abs(); positive weight → more arm-side run is better.
    weights = {
        "Fastball":       {"RelSpeed": 2.0, "InducedVertBreak": 2.5, "HorzBreak": 0.3, "Extension": 0.5, "VertApprAngle": 2.5, "SpinRate": 1.0},
        "Sinker":         {"RelSpeed": 2.5, "InducedVertBreak": -0.5, "HorzBreak": 1.5, "Extension": 0.5, "VertApprAngle": -1.5, "SpinRate": 0.8},
        "Cutter":         {"RelSpeed": 0.8, "InducedVertBreak": 0.3, "HorzBreak": -1.5, "Extension": -1.0, "VertApprAngle": -0.5, "SpinRate": 2.0},
        "Slider":         {"RelSpeed": 1.0, "InducedVertBreak": -0.5, "HorzBreak": 1.0, "Extension": 0.3, "VertApprAngle": -2.5, "SpinRate": 1.5},
        "Curveball":      {"RelSpeed": 1.5, "InducedVertBreak": -1.5, "HorzBreak": -1.5, "Extension": -1.5, "VertApprAngle": -2.0, "SpinRate": 0.5},
        "Changeup":       {"RelSpeed": 0.5, "InducedVertBreak": 1.5, "HorzBreak": 1.0, "Extension": 0.5, "VertApprAngle": -2.5, "SpinRate": 1.0},
        "Sweeper":        {"RelSpeed": 1.5, "InducedVertBreak": -1.0, "HorzBreak": 2.0, "Extension": 0.8, "VertApprAngle": -1.5, "SpinRate": 0.5},
        "Splitter":       {"RelSpeed": 1.0, "InducedVertBreak": -2.0, "HorzBreak": 0.5, "Extension": 1.0, "VertApprAngle": -2.0, "SpinRate": -0.3},
        "Knuckle Curve":  {"RelSpeed": 1.5, "InducedVertBreak": -1.5, "HorzBreak": -1.5, "Extension": -1.5, "VertApprAngle": -2.0, "SpinRate": 0.5},
    }

    base = data.dropna(subset=["RelSpeed", "TaggedPitchType"]).copy()
    metrics = ["RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate", "VertApprAngle", "Extension"]

    # Determine pitcher handedness sign for HorzBreak:
    # RHP → positive HB = arm-side run; LHP → negative HB = arm-side run.
    # We normalise to abs(HB) so comparisons are hand-agnostic.
    throws = pdf["PitcherThrows"].mode()
    is_lhp = (throws.iloc[0] == "Left") if len(throws) > 0 else False
    hb_sign = -1.0 if is_lhp else 1.0  # multiplier to convert raw → abs convention

    # Pre-compute baseline per pitch type using abs(HorzBreak)
    base["_AbsHB"] = base["HorzBreak"].abs() if "HorzBreak" in base.columns else np.nan
    baseline = {}
    for pt, grp in base.groupby("TaggedPitchType"):
        stats = {}
        for m in metrics:
            col = "_AbsHB" if m == "HorzBreak" else m
            if col in grp.columns:
                vals = grp[col].astype(float).dropna()
                if len(vals) >= 10:
                    stats[m] = {"mean": vals.mean(), "std": vals.std(),
                                "p75": np.percentile(vals, 75),
                                "p25": np.percentile(vals, 25)}
        baseline[pt] = stats

    # Build tunnel partner map for cross-referencing
    tunnel_partners = {}
    if isinstance(tunnel_df, pd.DataFrame) and not tunnel_df.empty:
        for _, trow in tunnel_df.iterrows():
            a, b = trow["Pitch A"], trow["Pitch B"]
            tunnel_partners.setdefault(a, []).append((b, trow.get("Grade", "-"), trow.get("Tunnel Score", np.nan)))
            tunnel_partners.setdefault(b, []).append((a, trow.get("Grade", "-"), trow.get("Tunnel Score", np.nan)))

    recommendations = []
    for pt in sorted(pdf["TaggedPitchType"].unique()):
        pt_d = pdf[pdf["TaggedPitchType"] == pt]
        if len(pt_d) < 10:
            continue
        w = weights.get(pt, {})
        bstats = baseline.get(pt, {})
        if not w or not bstats:
            continue

        # For each metric, compute how far below the 75th pctl the pitcher is
        # in the "good" direction
        gaps = []
        for m in metrics:
            if m not in w or m not in bstats:
                continue
            bs = bstats[m]
            if bs["std"] == 0 or pd.isna(bs["std"]):
                continue

            # For HorzBreak, use absolute value so LHP/RHP are on the same scale
            if m == "HorzBreak":
                pitcher_val = pt_d[m].astype(float).dropna().abs().mean()
            else:
                pitcher_val = pt_d[m].astype(float).dropna().mean()
            if pd.isna(pitcher_val):
                continue

            weight_sign = 1 if w[m] > 0 else -1

            # Percentile of pitcher vs abs-HB baseline (or normal baseline)
            if m == "HorzBreak":
                all_vals = base[base["TaggedPitchType"] == pt]["_AbsHB"].astype(float).dropna()
            else:
                all_vals = base[base["TaggedPitchType"] == pt][m].astype(float).dropna()
            if len(all_vals) < 10:
                continue
            pctl = _pctile(all_vals, pitcher_val, kind="rank")
            good_pctl = pctl if weight_sign > 0 else (100 - pctl)

            # Target = 75th percentile in the good direction
            if weight_sign > 0:
                target = bs["p75"]
            else:
                target = bs["p25"]

            # Only flag if pitcher is below 60th pctl in good direction
            if good_pctl < 60:
                gaps.append({
                    "metric": m,
                    "good_pctl": good_pctl,
                    "current": pitcher_val,
                    "target": target,
                    "weight_sign": weight_sign,
                    "abs_weight": abs(w[m]),
                })

        # Sort by worst percentile, weighted by importance
        gaps.sort(key=lambda g: g["good_pctl"] - g["abs_weight"] * 5)

        # Take top 2 recommendations
        for g in gaps[:2]:
            m = g["metric"]
            label = _METRIC_LABELS.get(m, m)
            unit = _METRIC_UNITS.get(m, "")
            current_abs = g["current"]  # already abs for HB
            target_abs = g["target"]
            ws = g["weight_sign"]

            if ws > 0:
                delta_val = target_abs - current_abs
                direction = f"more {label.lower()}" if delta_val > 0 else f"maintain {label.lower()}"
            else:
                delta_val = current_abs - target_abs
                direction = f"{'steeper' if m == 'VertApprAngle' else 'more'} {label.lower()}" if delta_val > 0 else f"reduce {label.lower()}"

            # For display: convert HorzBreak back to the pitcher's sign convention
            if m == "HorzBreak":
                display_current = current_abs * hb_sign
                display_target = target_abs * hb_sign
                # Delta is always in magnitude terms (positive = more break)
                delta_display = target_abs - current_abs
            else:
                display_current = current_abs
                display_target = target_abs
                delta_display = delta_val

            # Cross-reference with tunnel partners
            tunnel_benefit = ""
            partners = tunnel_partners.get(pt, [])
            if partners:
                # Find a partner where improving this metric would help
                for partner_pt, tgrade, tscore in partners:
                    if tgrade in ("D", "F", "C"):
                        if m == "InducedVertBreak":
                            tunnel_benefit = f"Improves tunnel with {partner_pt} (currently grade {tgrade})"
                        elif m == "HorzBreak":
                            tunnel_benefit = f"Creates better separation from {partner_pt} (currently grade {tgrade})"
                        elif m == "RelSpeed":
                            tunnel_benefit = f"Affects velo gap with {partner_pt} (currently grade {tgrade})"
                        elif m == "VertApprAngle":
                            tunnel_benefit = f"Helps deception against {partner_pt} (currently grade {tgrade})"
                        if tunnel_benefit:
                            break

            recommendations.append({
                "pitch": pt,
                "metric": m,
                "label": label,
                "current": round(display_current, 1),
                "target": round(display_target, 1),
                "delta": f"{'+' if delta_display > 0 else ''}{delta_display:.1f} {unit}",
                "direction": direction,
                "unit": unit,
                "good_pctl": round(g["good_pctl"], 0),
                "tunnel_benefit": tunnel_benefit,
            })

    return recommendations


def _pitch_lab_page(data, pitcher, season_filter, pdf, stuff_df, pr, all_pitcher_stats):
    """Render the Pitch Lab tab with arsenal recommendations, tunnel roadmap,
    sequencing playbook, and pitch-specific deep dives."""

    has_stuff = stuff_df is not None and "StuffPlus" in stuff_df.columns and not stuff_df.empty
    if not has_stuff:
        st.info("Stuff+ data not available — some sections will be limited.")

    # Pre-compute shared data
    tunnel_pop = build_tunnel_population_pop()
    tunnel_df = _compute_tunnel_score(pdf, tunnel_pop=tunnel_pop)
    cmd_df = _compute_command_plus(pdf, data)
    cmd_map = dict(zip(cmd_df["Pitch"], cmd_df["Command+"])) if not cmd_df.empty else {}

    pitch_types = sorted(pdf["TaggedPitchType"].unique())

    # ═══════════════════════════════════════════
    # SECTION A: Arsenal Overview with Improvement Targets
    # ═══════════════════════════════════════════
    section_header("Arsenal Overview & Improvement Targets")
    st.caption("Current pitch profiles compared to database averages. Recommendations based on Stuff+ weight directions and tunnel partners.")

    recommendations = _compute_pitch_recommendations(pdf, data, tunnel_df)
    rec_by_pitch = {}
    for r in recommendations:
        rec_by_pitch.setdefault(r["pitch"], []).append(r)

    for pt in pitch_types:
        pt_d = pdf[pdf["TaggedPitchType"] == pt]
        if len(pt_d) < 10:
            continue
        color = PITCH_COLORS.get(pt, "#888")

        # Compute metrics
        velo = pt_d["RelSpeed"].mean()
        ivb = pt_d["InducedVertBreak"].mean() if "InducedVertBreak" in pt_d.columns else np.nan
        hb = pt_d["HorzBreak"].mean() if "HorzBreak" in pt_d.columns else np.nan
        spin = pt_d["SpinRate"].mean() if "SpinRate" in pt_d.columns else np.nan
        vaa = pt_d["VertApprAngle"].mean() if "VertApprAngle" in pt_d.columns else np.nan
        ext = pt_d["Extension"].mean() if "Extension" in pt_d.columns else np.nan
        stuff_val = stuff_df[stuff_df["TaggedPitchType"] == pt]["StuffPlus"].mean() if has_stuff else np.nan
        cmd_val = cmd_map.get(pt, np.nan)

        # Header bar
        stuff_str = f"Stuff+ {stuff_val:.0f}" if not pd.isna(stuff_val) else ""
        cmd_str = f"Cmd+ {cmd_val:.0f}" if not pd.isna(cmd_val) else ""
        badges = " &middot; ".join(filter(None, [stuff_str, cmd_str]))
        st.markdown(
            f'<div style="padding:8px 14px;border-radius:8px;border-left:5px solid {color};'
            f'background:{color}12;margin:8px 0 4px 0;">'
            f'<span style="font-size:16px;font-weight:bold;color:{color};">{pt}</span>'
            f'<span style="font-size:12px;color:#555;margin-left:12px;">{badges}</span>'
            f'</div>', unsafe_allow_html=True)

        # Profile metrics in columns
        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
        def _metric_cell(col, label, val, fmt=".1f", unit=""):
            with col:
                v_str = f"{val:{fmt}}{unit}" if not pd.isna(val) else "-"
                st.markdown(f'<div style="text-align:center;"><div style="font-size:11px;color:#888;">{label}</div>'
                            f'<div style="font-size:16px;font-weight:bold;">{v_str}</div></div>',
                            unsafe_allow_html=True)
        _metric_cell(mc1, "Velo", velo, ".1f", " mph")
        _metric_cell(mc2, "IVB", ivb, ".1f", '"')
        _metric_cell(mc3, "HB", hb, ".1f", '"')
        _metric_cell(mc4, "Spin", spin, ".0f", " rpm")
        _metric_cell(mc5, "VAA", vaa, ".1f", "°")
        _metric_cell(mc6, "Ext", ext, ".1f", " ft")

        # Recommendations for this pitch
        recs = rec_by_pitch.get(pt, [])
        if recs:
            for rec in recs:
                pctl_str = f"{rec['good_pctl']:.0f}th pctl"
                tun_str = f" — {rec['tunnel_benefit']}" if rec['tunnel_benefit'] else ""
                st.markdown(
                    f'<div style="padding:4px 12px;margin:2px 0;font-size:12px;'
                    f'background:#fff8e1;border-radius:4px;border-left:3px solid #f59e0b;">'
                    f'<b>{rec["direction"].capitalize()}</b>: '
                    f'currently {rec["current"]} {rec["unit"]} ({pctl_str}), '
                    f'target {rec["target"]} {rec["unit"]} ({rec["delta"]})'
                    f'{tun_str}'
                    f'</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="padding:4px 12px;margin:2px 0;font-size:12px;'
                'background:#e8f5e9;border-radius:4px;border-left:3px solid #22c55e;">'
                'No major improvement targets — pitch metrics are solid relative to peers.'
                '</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    # SECTION B: Tunnel Improvement Roadmap
    # ═══════════════════════════════════════════
    st.markdown("---")
    section_header("Tunnel Improvement Roadmap")
    st.caption("All tunnel pairs ranked by priority. Worst grades first — focus fixes here.")

    if isinstance(tunnel_df, pd.DataFrame) and not tunnel_df.empty:
        # Sort worst first
        grade_order = {"F": 0, "D": 1, "C": 2, "B": 3, "A": 4}
        tdf_sorted = tunnel_df.copy()
        tdf_sorted["_grade_ord"] = tdf_sorted["Grade"].map(grade_order).fillna(2)
        tdf_sorted = tdf_sorted.sort_values("_grade_ord").drop(columns=["_grade_ord"])

        grade_colors = {"A": "#22c55e", "B": "#3b82f6", "C": "#f59e0b",
                        "D": "#f97316", "F": "#ef4444"}

        for _, row in tdf_sorted.iterrows():
            gc = grade_colors.get(row["Grade"], "#888")
            commit_str = f'{row["Commit Sep (in)"]:.1f}' if "Commit Sep (in)" in row and not pd.isna(row.get("Commit Sep (in)")) else "-"
            highlight = "border-width:2px;" if row["Grade"] in ("F", "D") else ""

            st.markdown(
                f'<div style="padding:10px 14px;border-radius:8px;'
                f'border-left:4px solid {gc};{highlight}background:{gc}10;margin:4px 0;">'
                f'<span style="font-size:22px;font-weight:bold;color:{gc};">'
                f'{row["Grade"]}</span>'
                f'<span style="font-size:14px;font-weight:600;margin-left:10px;">'
                f'{row["Pitch A"]} + {row["Pitch B"]}</span>'
                f'<span style="font-size:12px;color:#666;margin-left:10px;">'
                f'Score: {row["Tunnel Score"]} &middot; Commit Sep: {commit_str} in</span>'
                f'</div>', unsafe_allow_html=True)

        # Release point overlay for each pair
        st.markdown("")
        st.caption("Release Point Overlays")
        rel_cols_needed = ["RelSide", "RelHeight", "TaggedPitchType"]
        has_rel = all(c in pdf.columns for c in rel_cols_needed)
        if has_rel:
            pairs_to_show = tdf_sorted.head(4)
            n_pairs = len(pairs_to_show)
            if n_pairs > 0:
                rel_cols = st.columns(min(n_pairs, 2))
                for pidx, (_, prow) in enumerate(pairs_to_show.iterrows()):
                    with rel_cols[pidx % 2]:
                        pa, pb = prow["Pitch A"], prow["Pitch B"]
                        da = pdf[pdf["TaggedPitchType"] == pa].dropna(subset=["RelSide", "RelHeight"])
                        db = pdf[pdf["TaggedPitchType"] == pb].dropna(subset=["RelSide", "RelHeight"])
                        if len(da) >= 3 and len(db) >= 3:
                            fig_rel = go.Figure()
                            ca = PITCH_COLORS.get(pa, "#888")
                            cb = PITCH_COLORS.get(pb, "#888")
                            fig_rel.add_trace(go.Scatter(
                                x=da["RelSide"], y=da["RelHeight"], mode="markers",
                                marker=dict(size=6, color=ca, opacity=0.6),
                                name=pa,
                            ))
                            fig_rel.add_trace(go.Scatter(
                                x=db["RelSide"], y=db["RelHeight"], mode="markers",
                                marker=dict(size=6, color=cb, opacity=0.6),
                                name=pb,
                            ))
                            fig_rel.update_layout(
                                height=250, title=f"{pa} vs {pb}",
                                xaxis_title="RelSide (ft)", yaxis_title="RelHeight (ft)",
                                xaxis=dict(scaleanchor="y"),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                            xanchor="center", x=0.5, font=dict(size=10)),
                                plot_bgcolor="white", paper_bgcolor="white",
                                font=dict(size=11, color="#1a1a2e", family="Inter, Arial, sans-serif"),
                                margin=dict(l=40, r=10, t=40, b=35),
                            )
                            st.plotly_chart(fig_rel, use_container_width=True,
                                            key=f"plab_rel_{pa}_{pb}")

        # Movement delta visualization
        st.caption("Movement Profiles (IVB vs HB)")
        mov_fig = make_movement_profile(pdf, height=380)
        if mov_fig is not None:
            st.plotly_chart(mov_fig, use_container_width=True, key="plab_mov_profile")
    else:
        st.info("Not enough pitch variety to compute tunnel pairs.")

    # ═══════════════════════════════════════════
    # SECTION C: Sequencing Playbook
    # ═══════════════════════════════════════════
    st.markdown("---")
    section_header("Sequencing Playbook")
    st.caption("Transition frequencies, sequence effectiveness, and count-state pitch selection.")

    pair_df = _compute_pitch_pair_results(pdf, data, tunnel_df=tunnel_df)

    # Transition matrix
    sort_cols_seq = [c for c in ["GameID", "Batter", "PAofInning", "PitchNo"] if c in pdf.columns]
    if len(sort_cols_seq) >= 2:
        pdf_seq = pdf.sort_values(sort_cols_seq).copy()
        pdf_seq["NextPitch"] = pdf_seq.groupby(["GameID", "Batter", "PAofInning"])["TaggedPitchType"].shift(-1)
        trans = pdf_seq.dropna(subset=["NextPitch"])
        if not trans.empty and len(trans["TaggedPitchType"].unique()) >= 2:
            tcol1, tcol2 = st.columns(2)
            with tcol1:
                st.markdown("**Pitch Transition Matrix**")
                matrix = pd.crosstab(trans["TaggedPitchType"], trans["NextPitch"], normalize="index") * 100
                fig_matrix = px.imshow(
                    matrix.round(1), text_auto=".0f", color_continuous_scale="RdBu_r",
                    labels=dict(x="Next Pitch", y="Current Pitch", color="%"),
                    aspect="auto",
                )
                fig_matrix.update_layout(height=350, **CHART_LAYOUT)
                st.plotly_chart(fig_matrix, use_container_width=True, key="plab_trans")

            with tcol2:
                if not pair_df.empty:
                    st.markdown("**Whiff% by Sequence**")
                    whiff_pivot = pair_df.pivot_table(index="Setup Pitch", columns="Follow Pitch",
                                                      values="Whiff%", aggfunc="first")
                    if not whiff_pivot.empty:
                        fig_whiff = px.imshow(
                            whiff_pivot.fillna(0).round(1), text_auto=".0f",
                            color_continuous_scale="YlOrRd",
                            labels=dict(x="Follow-Up Pitch", y="Setup Pitch", color="Whiff%"),
                            aspect="auto",
                        )
                        fig_whiff.update_layout(height=350, **CHART_LAYOUT)
                        st.plotly_chart(fig_whiff, use_container_width=True, key="plab_whiff_hm")

    # Count-state pitch selection
    st.markdown("**Count-State Pitch Selection**")
    counts_of_interest = [("0", "0"), ("0", "2"), ("1", "2"), ("2", "0"), ("3", "1"), ("3", "2")]
    count_rows = []
    for b, s in counts_of_interest:
        count_data = pdf[(pdf["Balls"].astype(str) == b) & (pdf["Strikes"].astype(str) == s)]
        if len(count_data) >= 3:
            usage = count_data["TaggedPitchType"].value_counts(normalize=True) * 100
            for pt_name, pct in usage.items():
                count_rows.append({"Count": f"{b}-{s}", "Pitch": pt_name, "Usage%": round(pct, 1)})
    if count_rows:
        count_df = pd.DataFrame(count_rows)
        fig_count = px.bar(
            count_df, x="Count", y="Usage%", color="Pitch",
            color_discrete_map=PITCH_COLORS, barmode="stack",
        )
        fig_count.update_layout(
            height=350, yaxis_title="Usage %", xaxis_title="Count",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            **CHART_LAYOUT,
        )
        st.plotly_chart(fig_count, use_container_width=True, key="plab_count_sel")
    else:
        st.info("Not enough count-state data.")

    # ── Build ps_items for sequence computation (shared) ──
    ps_items = []
    total_p = len(pdf)
    for pt in pitch_types:
        pt_d = pdf[pdf["TaggedPitchType"] == pt]
        if len(pt_d) < 10:
            continue
        usage = len(pt_d) / total_p * 100
        sw = pt_d[pt_d["PitchCall"].isin(SWING_CALLS)]
        whiff_r = len(pt_d[pt_d["PitchCall"] == "StrikeSwinging"]) / max(len(sw), 1) * 100
        out_z = ~in_zone_mask(pt_d)
        chase_sw = pt_d[out_z & pt_d["PitchCall"].isin(SWING_CALLS)]
        chase_r = len(chase_sw) / max(out_z.sum(), 1) * 100
        stuff_avg = np.nan
        if has_stuff:
            st_v = stuff_df[stuff_df["TaggedPitchType"] == pt]["StuffPlus"]
            if len(st_v) > 0:
                stuff_avg = st_v.mean()
        vals = []
        if not pd.isna(stuff_avg):
            vals.append(("s", min(max((stuff_avg - 70) / 60 * 100, 0), 100)))
        if whiff_r > 0:
            vals.append(("w", min(whiff_r / 50 * 100, 100)))
        if chase_r > 0:
            vals.append(("c", min(chase_r / 40 * 100, 100)))
        if vals:
            wts = {"s": 0.5, "w": 0.3, "c": 0.2}
            tw = sum(wts[k] for k, _ in vals)
            comp = sum(wts[k] * v for k, v in vals) / tw if tw > 0 else 50
        else:
            comp = 50
        ps_items.append((pt, {"count": len(pt_d), "usage": usage, "score": comp,
                               "our_whiff": whiff_r, "our_chase": chase_r,
                               "velo": pt_d["RelSpeed"].mean(),
                               "eff_velo": np.nan}))
    ps_items.sort(key=lambda x: x[1]["score"], reverse=True)

    # Helper: render a sequence card
    grade_colors_seq = {"A": "#22c55e", "B": "#3b82f6", "C": "#f59e0b",
                        "D": "#f97316", "F": "#ef4444"}

    def _render_seq_card(seq, tunnel_df_ref, key_prefix=""):
        pitches_in_seq = [seq.get(f"p{i}") for i in range(1, 5) if seq.get(f"p{i}")]
        tunnel_strs = []
        for i in range(len(pitches_in_seq) - 1):
            t_key = f"t{i+1}{i+2}"
            t_val = seq.get(t_key, np.nan)
            if not pd.isna(t_val):
                _, g = _lookup_tunnel(pitches_in_seq[i], pitches_in_seq[i+1], tunnel_df_ref)
            else:
                g = "-"
            gc = grade_colors_seq.get(g, "#888")
            tunnel_strs.append(f'T<sub>{i+1}{i+2}</sub>: <span style="color:{gc};font-weight:bold;">{g}</span>')
        tunnels_html = " &middot; ".join(tunnel_strs)
        st.markdown(
            f'<div style="text-align:center;padding:12px;border-radius:8px;'
            f'border:1px solid #ddd;background:#f9f9f9;margin:2px;">'
            f'<div style="font-size:14px;font-weight:bold;">{seq["seq"]}</div>'
            f'<div style="font-size:18px;font-weight:bold;color:#1a1a2e;">'
            f'Score: {seq["score"]:.0f}</div>'
            f'<div style="font-size:11px;color:#666;margin-top:4px;">{tunnels_html}</div>'
            f'</div>', unsafe_allow_html=True)

    # Helper: render location heatmaps for pitches in a sequence
    def _render_seq_locations(pitches_list, pdf_ref, key_prefix=""):
        n = len(pitches_list)
        loc_cols = st.columns(n)
        for i, pt_name in enumerate(pitches_list):
            with loc_cols[i]:
                pt_sub = pdf_ref[pdf_ref["TaggedPitchType"] == pt_name].dropna(
                    subset=["PlateLocSide", "PlateLocHeight"])
                color = PITCH_COLORS.get(pt_name, "#888")
                st.caption(f"{pt_name}")
                if len(pt_sub) >= 5:
                    # Show CSW locations (called strikes + whiffs = best outcomes)
                    csw = pt_sub[pt_sub["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"])]
                    plot_data = csw if len(csw) >= 3 else pt_sub
                    fig_loc = go.Figure(go.Histogram2d(
                        x=plot_data["PlateLocSide"], y=plot_data["PlateLocHeight"],
                        nbinsx=8, nbinsy=8, colorscale=[[0, "white"], [1, color]],
                        showscale=False,
                    ))
                    add_strike_zone(fig_loc)
                    fig_loc.update_layout(
                        xaxis=dict(range=[-2.5, 2.5], title="", showticklabels=False),
                        yaxis=dict(range=[0, 5], title="", showticklabels=False),
                        height=200, margin=dict(l=5, r=5, t=5, b=5),
                        plot_bgcolor="white", paper_bgcolor="white",
                    )
                    st.plotly_chart(fig_loc, use_container_width=True,
                                    key=f"{key_prefix}_loc_{pt_name}_{i}")
                else:
                    st.info("< 5 pitches")

    # Top 3-pitch sequences with tunnel grades
    if isinstance(tunnel_df, pd.DataFrame) and not tunnel_df.empty:
        top_seqs = _build_3pitch_sequences(ps_items, {}, tunnel_df, pair_df)
        if top_seqs:
            st.markdown("**Top 3-Pitch Sequences**")
            n_show = min(len(top_seqs), 3)
            seq_cols = st.columns(n_show)
            for idx, seq in enumerate(top_seqs[:n_show]):
                with seq_cols[idx]:
                    _render_seq_card(seq, tunnel_df)
            # Location heatmaps for the best 3-pitch sequence
            best3 = top_seqs[0]
            st.caption(f"Best locations for: {best3['seq']}")
            _render_seq_locations([best3["p1"], best3["p2"], best3["p3"]], pdf, "seq3")

    # Top 4-pitch sequences
    if isinstance(tunnel_df, pd.DataFrame) and not tunnel_df.empty and len(ps_items) >= 3:
        st.markdown("**Top 4-Pitch Sequences**")
        top3 = _build_3pitch_sequences(ps_items, {}, tunnel_df, pair_df)
        four_seqs = []
        if top3:
            used_combos = set()
            for s3 in top3[:5]:
                p1, p2, p3 = s3["p1"], s3["p2"], s3["p3"]
                for ext_pt, ext_stats in ps_items:
                    if ext_pt == p3:
                        continue
                    combo = (p1, p2, p3, ext_pt)
                    if combo in used_combos:
                        continue
                    t34_score, t34_grade = _lookup_tunnel(p3, ext_pt, tunnel_df)
                    if t34_grade == "-":
                        continue
                    ext_score = ext_stats["score"]
                    seq_score = s3["score"] * 0.6 + ext_score * 0.2
                    if not pd.isna(t34_score):
                        seq_score += t34_score * 0.2
                    four_seqs.append({
                        "seq": f"{p1} → {p2} → {p3} → {ext_pt}",
                        "p1": p1, "p2": p2, "p3": p3, "p4": ext_pt,
                        "score": seq_score,
                        "t12": s3.get("t12", np.nan),
                        "t23": s3.get("t23", np.nan),
                        "t34": t34_score,
                    })
                    used_combos.add(combo)
            four_seqs.sort(key=lambda x: x["score"], reverse=True)
        if four_seqs:
            n_show4 = min(len(four_seqs), 3)
            seq4_cols = st.columns(n_show4)
            for idx, seq in enumerate(four_seqs[:n_show4]):
                with seq4_cols[idx]:
                    _render_seq_card(seq, tunnel_df)
            # Location heatmaps for the best 4-pitch sequence
            best4 = four_seqs[0]
            st.caption(f"Best locations for: {best4['seq']}")
            _render_seq_locations([best4["p1"], best4["p2"], best4["p3"], best4["p4"]], pdf, "seq4")

    # ── Get Back in the Count: best pitches at 1-0 and 2-0 ──
    st.markdown("**Get Back in the Count (1-0, 2-0)**")
    st.caption("Best pitches to throw when behind — ranked by CSW% at that count.")
    for count_label, b_str, s_str in [("1-0", "1", "0"), ("2-0", "2", "0")]:
        count_d = pdf[(pdf["Balls"].astype(str) == b_str) & (pdf["Strikes"].astype(str) == s_str)]
        if len(count_d) < 5:
            st.info(f"Not enough data at {count_label}.")
            continue
        # Rank pitches by CSW% at this count
        ct_rows = []
        for cpt in count_d["TaggedPitchType"].unique():
            cpt_d = count_d[count_d["TaggedPitchType"] == cpt]
            if len(cpt_d) < 3:
                continue
            csw_ct = cpt_d["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"]).mean() * 100
            usage_ct = len(cpt_d) / len(count_d) * 100
            ct_rows.append({"Pitch": cpt, "CSW%": round(csw_ct, 1),
                            "Usage%": round(usage_ct, 1), "N": len(cpt_d)})
        if not ct_rows:
            continue
        ct_df = pd.DataFrame(ct_rows).sort_values("CSW%", ascending=False)
        st.markdown(f"**{count_label}**")
        # Show top pitches as inline badges
        badge_parts = []
        for _, cr in ct_df.head(3).iterrows():
            pc = PITCH_COLORS.get(cr["Pitch"], "#888")
            badge_parts.append(
                f'<span style="display:inline-block;padding:4px 10px;border-radius:12px;'
                f'background:{pc}20;border:1px solid {pc};margin:2px;font-size:12px;">'
                f'<b style="color:{pc};">{cr["Pitch"]}</b> '
                f'CSW {cr["CSW%"]:.0f}% &middot; {cr["Usage%"]:.0f}% used</span>')
        st.markdown(" ".join(badge_parts), unsafe_allow_html=True)
        # Location heatmaps for top 2 pitches at this count
        top_ct_pitches = ct_df.head(2)["Pitch"].tolist()
        gbc_cols = st.columns(len(top_ct_pitches))
        for ci, cpt_name in enumerate(top_ct_pitches):
            with gbc_cols[ci]:
                cpt_sub = count_d[count_d["TaggedPitchType"] == cpt_name].dropna(
                    subset=["PlateLocSide", "PlateLocHeight"])
                color_c = PITCH_COLORS.get(cpt_name, "#888")
                if len(cpt_sub) >= 3:
                    fig_gbc = go.Figure(go.Histogram2d(
                        x=cpt_sub["PlateLocSide"], y=cpt_sub["PlateLocHeight"],
                        nbinsx=8, nbinsy=8, colorscale=[[0, "white"], [1, color_c]],
                        showscale=False,
                    ))
                    add_strike_zone(fig_gbc)
                    fig_gbc.update_layout(
                        xaxis=dict(range=[-2.5, 2.5], title="", showticklabels=False),
                        yaxis=dict(range=[0, 5], title="", showticklabels=False),
                        height=220, margin=dict(l=5, r=5, t=5, b=5),
                        plot_bgcolor="white", paper_bgcolor="white",
                    )
                    st.plotly_chart(fig_gbc, use_container_width=True,
                                    key=f"plab_gbc_{count_label}_{cpt_name}")
                    st.caption(f"{cpt_name} at {count_label}")
                else:
                    st.caption(f"{cpt_name}: < 3 pitches")

    # ═══════════════════════════════════════════
    # SECTION D: Pitch-Specific Deep Dive
    # ═══════════════════════════════════════════
    st.markdown("---")
    section_header("Pitch-Specific Deep Dive")
    sel_pitch = st.selectbox("Select Pitch", pitch_types, key="plab_deep_pitch")
    pt_data = pdf[pdf["TaggedPitchType"] == sel_pitch].copy()

    if len(pt_data) < 10:
        st.info("Not enough pitches of this type to analyze.")
    else:
        # Stuff+ distribution violin
        if has_stuff:
            dd1, dd2 = st.columns(2)
            with dd1:
                st.markdown("**Stuff+ Distribution**")
                pt_stuff = stuff_df[stuff_df["TaggedPitchType"] == sel_pitch]["StuffPlus"]
                if len(pt_stuff) >= 5:
                    fig_violin = go.Figure(go.Violin(
                        y=pt_stuff, box_visible=True, meanline_visible=True,
                        fillcolor=PITCH_COLORS.get(sel_pitch, "#888"),
                        line_color=PITCH_COLORS.get(sel_pitch, "#888"), opacity=0.7,
                        name=sel_pitch,
                    ))
                    fig_violin.add_hline(y=100, line_dash="dash", line_color="#888",
                                         annotation_text="Avg (100)")
                    fig_violin.update_layout(height=300, showlegend=False,
                                             yaxis_title="Stuff+", **CHART_LAYOUT)
                    st.plotly_chart(fig_violin, use_container_width=True, key="plab_dd_violin")
                else:
                    st.info("Not enough Stuff+ data for violin plot.")

            # Rolling Stuff+ trend
            with dd2:
                st.markdown("**Rolling Stuff+ Trend**")
                stuff_time = stuff_df[stuff_df["TaggedPitchType"] == sel_pitch].copy()
                if "Date" in stuff_time.columns:
                    stuff_time = stuff_time.dropna(subset=["Date"]).sort_values("Date")
                if len(stuff_time) >= 15:
                    window = min(25, len(stuff_time) // 2)
                    stuff_time["StuffRolling"] = stuff_time["StuffPlus"].rolling(window, min_periods=5).mean()
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        x=list(range(len(stuff_time))), y=stuff_time["StuffRolling"],
                        mode="lines", line=dict(color=PITCH_COLORS.get(sel_pitch, "#888"), width=2),
                        name=sel_pitch,
                    ))
                    fig_trend.add_hline(y=100, line_dash="dash", line_color="#888")
                    fig_trend.update_layout(height=300, showlegend=False,
                                             xaxis_title="Pitch #", yaxis_title="Stuff+ (rolling)",
                                             **CHART_LAYOUT)
                    st.plotly_chart(fig_trend, use_container_width=True, key="plab_dd_trend")
                else:
                    st.info("Not enough data for trend.")

        # Command+ zone quadrant grid
        if not cmd_df.empty:
            cmd_row = cmd_df[cmd_df["Pitch"] == sel_pitch]
            if not cmd_row.empty:
                cr = cmd_row.iloc[0]
                st.markdown(f"**Command+ Profile** — Command+ **{cr['Command+']:.0f}** "
                            f"| Zone% {cr['Zone%']:.1f} | Edge% {cr['Edge%']:.1f} "
                            f"| Chase% {cr['Chase%']:.1f}")

        # Whiff / Called Strike / Weak Contact heatmaps
        st.markdown("**Outcome Heatmaps**")
        hm1, hm2, hm3 = st.columns(3)
        with hm1:
            whiff_d = pt_data[pt_data["PitchCall"] == "StrikeSwinging"]
            st.caption(f"Whiff Locations ({len(whiff_d)})")
            if len(whiff_d) >= 3:
                fig_w = go.Figure(go.Histogram2d(
                    x=whiff_d["PlateLocSide"], y=whiff_d["PlateLocHeight"],
                    nbinsx=10, nbinsy=10, colorscale="YlOrRd", showscale=False,
                ))
                add_strike_zone(fig_w)
                fig_w.update_layout(xaxis=dict(range=[-2.5, 2.5], title=""),
                                    yaxis=dict(range=[0, 5], title=""),
                                    height=280, **CHART_LAYOUT)
                st.plotly_chart(fig_w, use_container_width=True, key="plab_dd_whiff")
            else:
                st.info("< 3 whiffs")
        with hm2:
            cs_d = pt_data[pt_data["PitchCall"] == "StrikeCalled"]
            st.caption(f"Called Strikes ({len(cs_d)})")
            if len(cs_d) >= 3:
                fig_cs = go.Figure(go.Histogram2d(
                    x=cs_d["PlateLocSide"], y=cs_d["PlateLocHeight"],
                    nbinsx=10, nbinsy=10, colorscale="Blues", showscale=False,
                ))
                add_strike_zone(fig_cs)
                fig_cs.update_layout(xaxis=dict(range=[-2.5, 2.5], title=""),
                                     yaxis=dict(range=[0, 5], title=""),
                                     height=280, **CHART_LAYOUT)
                st.plotly_chart(fig_cs, use_container_width=True, key="plab_dd_cs")
            else:
                st.info("< 3 called strikes")
        with hm3:
            ip_d = pt_data[(pt_data["PitchCall"] == "InPlay") & pt_data["ExitSpeed"].notna()]
            weak = ip_d[ip_d["ExitSpeed"] < 85]
            st.caption(f"Weak Contact ({len(weak)})")
            if len(weak) >= 3:
                fig_wk = go.Figure(go.Histogram2d(
                    x=weak["PlateLocSide"], y=weak["PlateLocHeight"],
                    nbinsx=10, nbinsy=10, colorscale="Greens", showscale=False,
                ))
                add_strike_zone(fig_wk)
                fig_wk.update_layout(xaxis=dict(range=[-2.5, 2.5], title=""),
                                     yaxis=dict(range=[0, 5], title=""),
                                     height=280, **CHART_LAYOUT)
                st.plotly_chart(fig_wk, use_container_width=True, key="plab_dd_weak")
            else:
                st.info("< 3 weak contacts")


# ──────────────────────────────────────────────
# PAGE: PITCHING (merged Pitcher Card + Pitch Design Lab)
# ──────────────────────────────────────────────
def page_pitching(data):
    pitching = filter_davidson(data, "pitcher")
    if pitching.empty:
        st.warning("No pitching data found.")
        return

    pitchers = sorted(pitching["Pitcher"].unique())
    c1, c2 = st.columns([2, 3])
    with c1:
        pitcher = st.selectbox("Select Pitcher", pitchers, format_func=display_name, key="pitching_pitcher")
    with c2:
        all_seasons = get_all_seasons()
        season_filter = st.multiselect("Season", all_seasons, default=all_seasons, key="pitching_season")

    all_pitcher_stats = compute_pitcher_stats_pop(season_filter=season_filter)
    if all_pitcher_stats.empty or pitcher not in all_pitcher_stats["Pitcher"].values:
        st.info("Not enough data for this pitcher.")
        return

    pr = all_pitcher_stats[all_pitcher_stats["Pitcher"] == pitcher].iloc[0]
    pdf_raw = pitching[(pitching["Pitcher"] == pitcher) & (pitching["Season"].isin(season_filter))]
    pdf = filter_minor_pitches(pdf_raw)
    if pdf.empty or len(pdf) < 20:
        st.warning("Not enough pitch data (need 20+).")
        return

    # Player header
    jersey = JERSEY.get(pitcher, "")
    pos = POSITION.get(pitcher, "")
    throws = safe_mode(pdf["PitcherThrows"], "")
    thr = {"Right": "R", "Left": "L"}.get(throws, throws)
    total_pitches = len(pdf)
    player_header(pitcher, jersey, pos,
                  f"{pos}  |  Throws: {thr}  |  Davidson Wildcats",
                  f"{total_pitches} pitches  |  {int(pr['PA'])} PA faced  |  "
                  f"Seasons: {', '.join(str(int(s)) for s in sorted(season_filter))}")

    # Compute Stuff+ for pitcher card
    stuff_df = _compute_stuff_plus(pdf)

    tab_card, tab_lab = st.tabs(["Pitcher Card", "Pitch Lab"])
    with tab_card:
        _pitcher_card_content(data, pitcher, season_filter, pdf, stuff_df, pr, all_pitcher_stats)
    with tab_lab:
        _pitch_lab_page(data, pitcher, season_filter, pdf, stuff_df, pr, all_pitcher_stats)


# ──────────────────────────────────────────────
# PAGE: CATCHER ANALYTICS
# ──────────────────────────────────────────────
def page_catcher(data):
    st.header("Catcher Analytics")
    catching = data[
        (data["CatcherTeam"] == DAVIDSON_TEAM_ID) & (data["Catcher"].isin(ROSTER_2026))
    ] if "Catcher" in data.columns and "CatcherTeam" in data.columns else pd.DataFrame()

    if catching.empty:
        st.warning("No catcher data found. Ensure the data has Catcher/CatcherTeam columns.")
        return

    catchers = sorted(catching["Catcher"].unique())
    c1, c2 = st.columns([2, 3])
    with c1:
        selected = st.selectbox("Select Catcher", catchers, key="cat_c")
    with c2:
        all_seasons = get_all_seasons()
        sel_seasons = st.multiselect("Season", all_seasons, default=all_seasons, key="cat_s")

    cdf = catching[(catching["Catcher"] == selected) & (catching["Season"].isin(sel_seasons))]
    if cdf.empty:
        st.info("No data for this catcher in selected seasons.")
        return

    jersey = JERSEY.get(selected, "")
    pos = POSITION.get(selected, "C")
    player_header(selected, jersey, pos,
                  f"{pos}  |  Davidson Wildcats",
                  f"{len(cdf)} pitches received  |  "
                  f"Seasons: {', '.join(str(int(s)) for s in sorted(sel_seasons))}")

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        # Pop Time
        section_header("Pop Time & Exchange")
        pop = cdf.dropna(subset=["PopTime"])
        exc = cdf.dropna(subset=["ExchangeTime"]) if "ExchangeTime" in cdf.columns else pd.DataFrame()

        if len(pop) > 0:
            # Get all catchers' avg pop time for context
            all_catchers_pop = data.dropna(subset=["PopTime"]) if "PopTime" in data.columns else pd.DataFrame()
            catcher_avgs = all_catchers_pop.groupby("Catcher")["PopTime"].mean() if not all_catchers_pop.empty else pd.Series()
            catcher_avgs = catcher_avgs[all_catchers_pop.groupby("Catcher").size() >= 5] if not all_catchers_pop.empty else pd.Series()

            p_pop = pop["PopTime"].mean()
            pop_pct = get_percentile(p_pop, catcher_avgs) if len(catcher_avgs) > 0 else np.nan
            pop_metrics = [("Avg Pop Time", p_pop, pop_pct, ".2f", False)]

            if len(exc) > 0:
                all_catchers_exc = data.dropna(subset=["ExchangeTime"]) if "ExchangeTime" in data.columns else pd.DataFrame()
                catcher_exc_avgs = all_catchers_exc.groupby("Catcher")["ExchangeTime"].mean() if not all_catchers_exc.empty else pd.Series()
                catcher_exc_avgs = catcher_exc_avgs[all_catchers_exc.groupby("Catcher").size() >= 5] if not all_catchers_exc.empty else pd.Series()
                p_exc = exc["ExchangeTime"].mean()
                exc_pct = get_percentile(p_exc, catcher_exc_avgs) if len(catcher_exc_avgs) > 0 else np.nan
                pop_metrics.append(("Avg Exchange", p_exc, exc_pct, ".2f", False))

            render_savant_percentile_section(pop_metrics, None)
            st.caption(f"Based on {len(pop)} recorded throws. Lower is better.")

            # Pop time distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=pop["PopTime"], nbinsx=20,
                marker_color="#cc0000", opacity=0.7,
            ))
            fig.add_vline(x=p_pop, line_dash="solid", line_color="#333",
                          annotation_text=f"Avg: {p_pop:.2f}s",
                          annotation_position="top right")
            if len(catcher_avgs) > 0:
                db_avg = catcher_avgs.mean()
                fig.add_vline(x=db_avg, line_dash="dash", line_color="#999",
                              annotation_text=f"DB Avg: {db_avg:.2f}s",
                              annotation_position="top left")
            fig.update_layout(
                xaxis_title="Pop Time (s)", yaxis_title="Count",
                height=300, **CHART_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True, key="cat_pop_dist")
        else:
            st.info("No pop time data recorded.")

    with col2:
        # Framing analysis
        section_header("Receiving / Framing")
        # Called strikes on pitches outside the zone
        loc_data = cdf.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        if len(loc_data) > 0:
            _iz = in_zone_mask(loc_data)
            out_zone_pitches = loc_data[~_iz]
            called_strikes_out = out_zone_pitches[out_zone_pitches["PitchCall"] == "StrikeCalled"]
            frame_rate = len(called_strikes_out) / max(len(out_zone_pitches), 1) * 100

            # Context: all catchers' framing rate
            all_loc = data.dropna(subset=["PlateLocSide", "PlateLocHeight"])
            if "Catcher" in all_loc.columns:
                all_in_zone = in_zone_mask(all_loc)
                all_out = all_loc[~all_in_zone]
                catcher_frame_rates = []
                for c, grp in all_out.groupby("Catcher"):
                    if len(grp) < 50:
                        continue
                    cs = grp[grp["PitchCall"] == "StrikeCalled"]
                    catcher_frame_rates.append(len(cs) / max(len(grp), 1) * 100)
                frame_pct = get_percentile(frame_rate, pd.Series(catcher_frame_rates)) if catcher_frame_rates else np.nan
            else:
                frame_pct = np.nan

            render_savant_percentile_section(
                [("Frame Rate", frame_rate, frame_pct, ".1f", True)], None,
            )
            st.caption(f"Called strikes on {len(out_zone_pitches)} out-of-zone pitches")

            # Framing heatmap — where does this catcher get extra strikes?
            if len(called_strikes_out) >= 3:
                fig = make_pitch_location_heatmap(called_strikes_out, "Framed Strikes", "#1a7a1a", height=350)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="cat_frame_hm")
        else:
            st.info("No location data for framing analysis.")

    # Catcher receiving by pitcher
    st.markdown("---")
    section_header("Performance by Pitcher Caught")
    pitcher_rows = []
    for pitcher, grp in cdf.groupby("Pitcher"):
        if len(grp) < 20:
            continue
        sw = grp[grp["PitchCall"].isin(SWING_CALLS)]
        wh = grp[grp["PitchCall"] == "StrikeSwinging"]
        loc = grp.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        in_z = in_zone_mask(loc)
        out_z = loc[~in_z]
        cs_out = out_z[out_z["PitchCall"] == "StrikeCalled"]
        pitcher_rows.append({
            "Pitcher": pitcher,
            "Pitches": len(grp),
            "Whiff%": round(len(wh) / max(len(sw), 1) * 100, 1),
            "Frame%": round(len(cs_out) / max(len(out_z), 1) * 100, 1) if len(out_z) > 0 else None,
        })
    if pitcher_rows:
        st.dataframe(pd.DataFrame(pitcher_rows).sort_values("Pitches", ascending=False),
                      use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────
# PAGE: TEAM OVERVIEW
# ──────────────────────────────────────────────
def page_team(data):
    _logo_path = os.path.join(_APP_DIR, "logo_real.png")
    _celeb_path = os.path.join(_APP_DIR, "regionalchamps_copy.jpg")

    # ── Hero Banner ──
    st.markdown("""
    <style>
    .wildcats-hero {
        background: #111 !important;
        border-radius: 12px;
        padding: 28px 32px 22px 32px;
        margin-bottom: 18px;
        border: 2px solid #cc0000;
        text-align: center;
    }
    .wildcats-hero .subtitle {
        font-size: 11px; letter-spacing: 5px; color: #ccc !important;
        font-weight: 500; margin-bottom: 2px;
    }
    .wildcats-hero .title {
        font-size: 38px; font-weight: 900; color: #cc0000 !important;
        letter-spacing: 3px; line-height: 1.15;
    }
    .wildcats-hero .tagline {
        font-size: 10px; letter-spacing: 3px; color: #999 !important;
        margin-top: 4px;
    }
    .stat-card {
        background: #111 !important; border-radius: 8px; padding: 12px 8px;
        text-align: center; border: 1px solid #333;
    }
    .stat-card .val { font-size: 22px; font-weight: 800; color: #fff !important; }
    .stat-card .lbl { font-size: 10px; color: #ccc !important; letter-spacing: 1px; text-transform: uppercase; }
    .leader-card {
        background: #111 !important; border-radius: 8px; padding: 12px 16px;
        margin: 6px 0; border: 1px solid #333;
        border-left: 4px solid #cc0000;
    }
    .leader-card .cat { font-size: 11px; color: #ccc !important; letter-spacing: 1px; text-transform: uppercase; }
    .leader-card .name { font-size: 17px; font-weight: 700; color: #fff !important; }
    .leader-card .stat { font-size: 14px; font-weight: 600; color: #cc0000 !important; }
    .leader-card .note { font-size: 11px; color: #aaa !important; }
    .leader-card-alt { border-left-color: #fff !important; }
    .leader-card-alt .stat { color: #fff !important; }
    .leader-card-grn { border-left-color: #cc0000 !important; }
    .leader-card-grn .stat { color: #cc0000 !important; }
    .logo-center { display: flex; align-items: center; justify-content: center; height: 100%; min-height: 140px; }
    </style>
    """, unsafe_allow_html=True)

    # Banner row: logo | title | celebration photo
    c_logo, c_title, c_photo = st.columns([1, 3, 1.5])
    with c_logo:
        if os.path.exists(_logo_path):
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.image(_logo_path, width=110)
    with c_title:
        st.markdown("""
        <div class="wildcats-hero">
            <div class="subtitle">DAVIDSON BASEBALL</div>
            <div class="title">W.I.L.D.C.A.T.S.</div>
            <div class="tagline">Wildcat Intelligence &amp; Live Data Computing for Advanced Trackman Statistics</div>
        </div>
        """, unsafe_allow_html=True)
    with c_photo:
        if os.path.exists(_celeb_path):
            st.image(_celeb_path, use_container_width=True)

    # ── Data setup ──
    dav_pitching = data[data["PitcherTeam"] == DAVIDSON_TEAM_ID]
    dav_batting = data[data["BatterTeam"] == DAVIDSON_TEAM_ID]
    dav_data = pd.concat([dav_pitching, dav_batting]).drop_duplicates(subset=["Date", "Pitcher", "Batter", "PitchNo", "Inning", "PAofInning"])
    latest_date = data["Date"].max()
    dav_dates = pd.concat([dav_pitching["Date"], dav_batting["Date"]]).dropna().dt.date.nunique()
    n_pitchers = len([p for p in ROSTER_2026 if p in dav_pitching["Pitcher"].values])
    n_hitters = len([b for b in ROSTER_2026 if b in dav_batting["Batter"].values])

    # ── Top-level tabs: Dashboard + Game Log ──
    tab_dash, tab_gl = st.tabs(["Dashboard", "Game Log"])

    with tab_dash:
        _render_team_dashboard(data, dav_pitching, dav_batting, dav_data, latest_date, dav_dates, n_pitchers, n_hitters)

    with tab_gl:
        _render_game_log(data)


def _render_game_log(data):
    """Game Log tab inside Team Overview."""
    dav = data[(data["PitcherTeam"] == DAVIDSON_TEAM_ID) | (data["BatterTeam"] == DAVIDSON_TEAM_ID)]
    if dav.empty:
        st.warning("No data.")
        return
    games = dav.groupby(["Date", "GameID"]).agg(
        Stadium=("Stadium", "first"), Home=("HomeTeam", "first"),
        Away=("AwayTeam", "first"), Pitches=("PitchNo", "count"),
    ).reset_index().sort_values("Date", ascending=False)
    st.dataframe(games[["Date", "Away", "Home", "Stadium", "Pitches"]], use_container_width=True, hide_index=True)

    opts = games["GameID"].tolist()
    if opts:
        _game_labels = {row["GameID"]: f"{row['Date'].strftime('%Y-%m-%d')} {row['Away']} @ {row['Home']}"
                        for _, row in games.iterrows()}
        sel = st.selectbox("Drill into game", opts,
                           format_func=lambda g: _game_labels.get(g, str(g)), key="to_gl_game")
        gd = dav[dav["GameID"] == sel]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Davidson Pitching**")
            gp = gd[gd["PitcherTeam"] == DAVIDSON_TEAM_ID]
            for p in gp["Pitcher"].unique():
                sub = gp[gp["Pitcher"] == p]
                st.markdown(f"_{display_name(p)}_ - {len(sub)} pitches")
                s = sub.groupby("TaggedPitchType").agg(
                    N=("RelSpeed", "count"), Velo=("RelSpeed", "mean"), Spin=("SpinRate", "mean")
                ).reset_index()
                s["Velo"] = s["Velo"].round(1)
                s["Spin"] = s["Spin"].round(0)
                st.dataframe(s, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Davidson Hitting**")
            gh = gd[gd["BatterTeam"] == DAVIDSON_TEAM_ID]
            ip = gh[gh["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            for b in ip["Batter"].unique():
                sub = ip[ip["Batter"] == b]
                st.markdown(f"_{display_name(b)}_ - {len(sub)} BIP, Avg: {sub['ExitSpeed'].mean():.1f}, "
                            f"Max: {sub['ExitSpeed'].max():.1f}")


def _render_team_dashboard(data, dav_pitching, dav_batting, dav_data, latest_date, dav_dates, n_pitchers, n_hitters):
    """Dashboard tab inside Team Overview."""
    # ── Quick Stats Row ──
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.markdown(f'<div class="stat-card"><div class="val">{len(data):,}</div><div class="lbl">Total Pitches</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="stat-card"><div class="val">{dav_dates}</div><div class="lbl">Davidson Games</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="stat-card"><div class="val">{n_pitchers}</div><div class="lbl">Rostered Arms</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown(f'<div class="stat-card"><div class="val">{n_hitters}</div><div class="lbl">Rostered Bats</div></div>', unsafe_allow_html=True)
    with s5:
        _ld = latest_date.strftime("%b %d, %Y") if pd.notna(latest_date) else "—"
        st.markdown(f'<div class="stat-card"><div class="val" style="font-size:16px;">{_ld}</div><div class="lbl">Latest Data</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Weekly Leaders ──
    if pd.notna(latest_date):
        week_ago = latest_date - pd.Timedelta(days=7)
        recent = dav_data[dav_data["Date"] > week_ago]
    else:
        recent = pd.DataFrame()

    section_header("This Week's Leaders")

    if len(recent) < 10:
        st.info("Not enough data in the last 7 days for weekly leaders. Showing full-season leaderboards below.")
    else:
        st.caption(f"{week_ago.strftime('%b %d')} – {latest_date.strftime('%b %d, %Y')}")
        col_wh, col_wp = st.columns(2)

        # ── Hitting Leaders ──
        with col_wh:
            st.markdown("#### :red[Hitting]")
            week_bat = recent[recent["BatterTeam"] == DAVIDSON_TEAM_ID].copy()
            week_inplay = week_bat[week_bat["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            if len(week_inplay) >= 5:
                # Hardest contact
                ev_leader = week_inplay.groupby("Batter")["ExitSpeed"].agg(["mean", "max", "count"]).reset_index()
                ev_leader = ev_leader[ev_leader["count"] >= 3].sort_values("mean", ascending=False)
                if len(ev_leader) > 0:
                    top = ev_leader.iloc[0]
                    st.markdown(f'<div class="leader-card">'
                                f'<div class="cat">Hardest Contact</div>'
                                f'<div class="name">{display_name(top["Batter"])}</div>'
                                f'<div class="stat">{top["mean"]:.1f} avg EV</div>'
                                f'<div class="note">{int(top["count"])} balls in play &middot; {top["max"]:.1f} max</div>'
                                f'</div>', unsafe_allow_html=True)

                # Barrel leader (only show if someone actually barreled a ball)
                _showed_barrel = False
                if "Angle" in week_inplay.columns:
                    wb = week_inplay.copy()
                    wb["is_barrel"] = is_barrel_mask(wb)
                    barrel_ct = wb.groupby("Batter")["is_barrel"].agg(["sum", "count"]).reset_index()
                    barrel_ct = barrel_ct[(barrel_ct["count"] >= 3) & (barrel_ct["sum"] > 0)].copy()
                    if len(barrel_ct) > 0:
                        barrel_ct["rate"] = barrel_ct["sum"] / barrel_ct["count"] * 100
                        barrel_ct = barrel_ct.sort_values("sum", ascending=False)
                        tb = barrel_ct.iloc[0]
                        st.markdown(f'<div class="leader-card leader-card-alt">'
                                    f'<div class="cat">Most Barrels</div>'
                                    f'<div class="name">{display_name(tb["Batter"])}</div>'
                                    f'<div class="stat">{int(tb["sum"])} barrels ({tb["rate"]:.0f}%)</div>'
                                    f'<div class="note">{int(tb["count"])} balls in play</div>'
                                    f'</div>', unsafe_allow_html=True)
                        _showed_barrel = True
                if not _showed_barrel and "Angle" in week_inplay.columns:
                    # Fallback: best sweet spot rate
                    wb2 = week_inplay.copy()
                    wb2["is_ss"] = wb2["Angle"].between(8, 32)
                    ss_ct = wb2.groupby("Batter")["is_ss"].agg(["sum", "count"]).reset_index()
                    ss_ct = ss_ct[ss_ct["count"] >= 3].copy()
                    if len(ss_ct) > 0:
                        ss_ct["rate"] = ss_ct["sum"] / ss_ct["count"] * 100
                        ss_ct = ss_ct.sort_values("rate", ascending=False)
                        ts = ss_ct.iloc[0]
                        st.markdown(f'<div class="leader-card leader-card-alt">'
                                    f'<div class="cat">Best Sweet Spot%</div>'
                                    f'<div class="name">{display_name(ts["Batter"])}</div>'
                                    f'<div class="stat">{ts["rate"]:.0f}% sweet spot</div>'
                                    f'<div class="note">{int(ts["count"])} balls in play</div>'
                                    f'</div>', unsafe_allow_html=True)

                # Max single hit
                if week_inplay["ExitSpeed"].notna().any():
                    max_ev_row = week_inplay.loc[week_inplay["ExitSpeed"].idxmax()]
                    st.markdown(f'<div class="leader-card leader-card-grn">'
                                f'<div class="cat">Hardest Single Hit</div>'
                                f'<div class="name">{display_name(max_ev_row["Batter"])}</div>'
                                f'<div class="stat">{max_ev_row["ExitSpeed"]:.1f} mph</div>'
                                f'</div>', unsafe_allow_html=True)
            else:
                st.caption("Not enough in-play data this week.")

        # ── Pitching Leaders ──
        with col_wp:
            st.markdown("#### :red[Pitching]")
            week_pit = recent[recent["PitcherTeam"] == DAVIDSON_TEAM_ID].copy()
            if len(week_pit) >= 10:
                # Whiff leader
                pit_swings = week_pit[week_pit["PitchCall"].isin(SWING_CALLS)]
                pit_whiffs = week_pit[week_pit["PitchCall"] == "StrikeSwinging"]
                whiff_by_p = pit_swings.groupby("Pitcher").size().reset_index(name="swings")
                whiff_ct = pit_whiffs.groupby("Pitcher").size().reset_index(name="whiffs")
                whiff_by_p = whiff_by_p.merge(whiff_ct, on="Pitcher", how="left").fillna(0)
                whiff_by_p["whiff_rate"] = whiff_by_p["whiffs"] / whiff_by_p["swings"] * 100
                whiff_by_p = whiff_by_p[whiff_by_p["swings"] >= 10].sort_values("whiff_rate", ascending=False)
                if len(whiff_by_p) > 0:
                    tw = whiff_by_p.iloc[0]
                    st.markdown(f'<div class="leader-card">'
                                f'<div class="cat">Highest Whiff Rate</div>'
                                f'<div class="name">{display_name(tw["Pitcher"])}</div>'
                                f'<div class="stat">{tw["whiff_rate"]:.0f}% whiff</div>'
                                f'<div class="note">{int(tw["swings"])} swings faced</div>'
                                f'</div>', unsafe_allow_html=True)

                # Velo leader
                fb_data = week_pit[week_pit["TaggedPitchType"].isin(["Fastball", "Sinker", "Cutter"])]
                if len(fb_data) > 0:
                    fb_velo = fb_data.groupby("Pitcher")["RelSpeed"].agg(["mean", "max", "count"]).reset_index()
                    fb_velo = fb_velo[fb_velo["count"] >= 5].sort_values("max", ascending=False)
                    if len(fb_velo) > 0:
                        tv = fb_velo.iloc[0]
                        st.markdown(f'<div class="leader-card leader-card-alt">'
                                    f'<div class="cat">Top Velocity</div>'
                                    f'<div class="name">{display_name(tv["Pitcher"])}</div>'
                                    f'<div class="stat">{tv["max"]:.1f} max &middot; {tv["mean"]:.1f} avg</div>'
                                    f'<div class="note">{int(tv["count"])} fastballs thrown</div>'
                                    f'</div>', unsafe_allow_html=True)

                # Lowest EV against
                pit_inplay = recent[(recent["PitcherTeam"] == DAVIDSON_TEAM_ID) & (recent["PitchCall"] == "InPlay")]
                pit_ev = pit_inplay.dropna(subset=["ExitSpeed"]).groupby("Pitcher")["ExitSpeed"].agg(["mean", "count"]).reset_index()
                pit_ev = pit_ev[pit_ev["count"] >= 3].sort_values("mean")
                if len(pit_ev) > 0:
                    te = pit_ev.iloc[0]
                    st.markdown(f'<div class="leader-card leader-card-grn">'
                                f'<div class="cat">Weakest Contact Allowed</div>'
                                f'<div class="name">{display_name(te["Pitcher"])}</div>'
                                f'<div class="stat">{te["mean"]:.1f} avg EV against</div>'
                                f'<div class="note">{int(te["count"])} balls in play</div>'
                                f'</div>', unsafe_allow_html=True)
            else:
                st.caption("Not enough pitching data this week.")

    st.markdown("")

    # ── Full Season Leaderboards ──
    section_header("Season Leaderboards")
    all_seasons = get_all_seasons()
    sel = st.multiselect("Season", all_seasons, default=all_seasons, key="to_s")

    tab_h, tab_p = st.tabs(["Hitting Leaderboard", "Pitching Leaderboard"])

    with tab_h:
        bs = compute_batter_stats_pop(season_filter=sel)
        dav_h = bs[(bs["BatterTeam"] == DAVIDSON_TEAM_ID) & (bs["Batter"].isin(ROSTER_2026))].copy()
        if dav_h.empty:
            st.info("No hitting data.")
        else:
            d = dav_h[["Batter", "PA", "BBE", "AvgEV", "MaxEV", "HardHitPct", "BarrelPct",
                       "SweetSpotPct", "WhiffPct", "KPct", "BBPct", "ChasePct"]].sort_values("PA", ascending=False).copy()
            d["Batter"] = d["Batter"].apply(display_name)
            d.columns = ["Batter", "PA", "BBE", "Avg EV", "Max EV", "Hard%", "Barrel%",
                         "Sweet%", "Whiff%", "K%", "BB%", "Chase%"]
            for c in d.columns[3:]:
                d[c] = d[c].round(1)
            st.dataframe(d, use_container_width=True, hide_index=True)

    with tab_p:
        ps = compute_pitcher_stats_pop(season_filter=sel)
        dav_p = ps[(ps["PitcherTeam"] == DAVIDSON_TEAM_ID) & (ps["Pitcher"].isin(ROSTER_2026))].copy()
        if dav_p.empty:
            st.info("No pitching data.")
        else:
            d = dav_p[["Pitcher", "Pitches", "PA", "AvgFBVelo", "MaxFBVelo", "AvgSpin",
                       "WhiffPct", "KPct", "BBPct", "ZonePct", "ChasePct", "AvgEVAgainst", "GBPct"]].sort_values("Pitches", ascending=False).copy()
            d["Pitcher"] = d["Pitcher"].apply(display_name)
            d.columns = ["Pitcher", "Pitches", "PA", "Avg FB", "Max FB", "Spin",
                         "Whiff%", "K%", "BB%", "Zone%", "Chase%", "EV Ag", "GB%"]
            for c in d.columns[3:]:
                d[c] = d[c].round(1)
            st.dataframe(d, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────
# PAGE: PLAYER DEVELOPMENT
# ──────────────────────────────────────────────
def page_development(data):
    st.header("Player Development")
    role = st.radio("View", ["Pitcher", "Hitter"], horizontal=True)

    if role == "Pitcher":
        pdf = filter_davidson(data, "pitcher")
        if pdf.empty:
            st.warning("No data.")
            return
        player = st.selectbox("Pitcher", sorted(pdf["Pitcher"].unique()), key="dv_p")
        pdata = pdf[pdf["Pitcher"] == player]
        metric = st.selectbox("Metric", ["RelSpeed", "SpinRate", "InducedVertBreak", "HorzBreak",
                                          "Extension", "VertApprAngle", "EffectiveVelo"], key="dv_mp")
        pts = sorted(pdata["TaggedPitchType"].dropna().unique())
        sel_pt = st.multiselect("Pitch Types", pts, default=pts, key="dv_pt")
        pdata = pdata[pdata["TaggedPitchType"].isin(sel_pt)]
        if pdata[metric].notna().any():
            daily = pdata.groupby(["Date", "TaggedPitchType"])[metric].agg(["mean", "count"]).reset_index()
            daily.columns = ["Date", "PitchType", "Value", "Count"]
            fig = px.scatter(daily, x="Date", y="Value", color="PitchType", size="Count",
                             color_discrete_map=PITCH_COLORS, trendline="lowess",
                             labels={"Value": metric})
            fig.update_layout(height=400, **CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

            window = st.slider("Rolling window", 10, 100, 25, key="dv_wp")
            pdata_s = pdata.sort_values("Date")
            for pt in sel_pt:
                sub = pdata_s[pdata_s["TaggedPitchType"] == pt][metric].dropna()
                if len(sub) >= window:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=list(range(len(sub))), y=sub.rolling(window).mean(),
                                             name=pt, line=dict(color=PITCH_COLORS.get(pt, "#777"))))
                    fig.update_layout(title=f"{pt} Rolling {window} {metric}",
                                      xaxis_title="Pitch #", yaxis_title=metric, height=220,
                                      **CHART_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)
    else:
        hdf = filter_davidson(data, "batter")
        if hdf.empty:
            st.warning("No data.")
            return
        player = st.selectbox("Hitter", sorted(hdf["Batter"].unique()), key="dv_h")
        batted = hdf[(hdf["Batter"] == player) & (hdf["PitchCall"] == "InPlay")].dropna(subset=["ExitSpeed"])
        metric = st.selectbox("Metric", ["ExitSpeed", "Angle", "Distance"], key="dv_mh")
        if batted[metric].notna().any():
            daily = batted.groupby("Date")[metric].agg(["mean", "count"]).reset_index()
            daily.columns = ["Date", "Value", "Count"]
            fig = px.scatter(daily, x="Date", y="Value", size="Count", trendline="lowess",
                             labels={"Value": metric})
            fig.update_layout(height=350, **CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# PAGE: SCOUTING (TrueMedia-powered)
# ──────────────────────────────────────────────

def _hitter_attack_plan(rate, exit_d, pr, ht, hl):
    """Auto-generate 'How to Attack' text for a hitter based on TrueMedia data."""
    notes = []
    chase = rate.iloc[0].get("Chase%") if not rate.empty and "Chase%" in rate.columns else None
    if chase is not None and not pd.isna(chase) and chase > 30:
        notes.append(f"High chase rate ({chase:.1f}%) — expand off the plate and below the zone")
    elif chase is not None and not pd.isna(chase) and chase < 18:
        notes.append(f"Very disciplined ({chase:.1f}% chase) — must live in the zone, compete with strikes")
    k_pct = rate.iloc[0].get("K%") if not rate.empty and "K%" in rate.columns else None
    if k_pct is not None and not pd.isna(k_pct) and k_pct > 25:
        notes.append(f"Strikeout-prone ({k_pct:.1f}% K) — use putaway secondary pitches")
    bb_pct = rate.iloc[0].get("BB%") if not rate.empty and "BB%" in rate.columns else None
    if bb_pct is not None and not pd.isna(bb_pct) and bb_pct > 12:
        notes.append(f"Patient eye ({bb_pct:.1f}% BB) — don't nibble, throw strikes early")
    gb_pct = ht.iloc[0].get("Ground%") if not ht.empty and "Ground%" in ht.columns else None
    if gb_pct is not None and not pd.isna(gb_pct) and gb_pct < 30:
        notes.append(f"Low ground ball rate ({gb_pct:.1f}%) — fly-ball hitter, keep the ball down")
    ev = exit_d.iloc[0].get("ExitVel") if not exit_d.empty and "ExitVel" in exit_d.columns else None
    if ev is not None and not pd.isna(ev) and ev > 90:
        notes.append(f"Dangerous exit velo ({ev:.1f} mph) — do not groove fastballs")
    barrel = exit_d.iloc[0].get("Barrel%") if not exit_d.empty and "Barrel%" in exit_d.columns else None
    if barrel is not None and not pd.isna(barrel) and barrel > 12:
        notes.append(f"High barrel rate ({barrel:.1f}%) — pitch to contact with movement, not velocity")
    pull = hl.iloc[0].get("HPull%") if not hl.empty and "HPull%" in hl.columns else None
    if pull is not None and not pd.isna(pull) and pull > 50:
        notes.append(f"Pull-heavy ({pull:.1f}%) — attack away, shift infield towards pull side")
    if not notes:
        notes.append("No major exploitable tendencies identified. Pitch competitively and mix locations.")
    return notes


def _pitcher_attack_plan(trad, mov, pr, ht):
    """Auto-generate 'How to Attack' text for an opposing pitcher."""
    notes = []
    era = trad.iloc[0].get("ERA") if not trad.empty and "ERA" in trad.columns else None
    fip = trad.iloc[0].get("FIP") if not trad.empty and "FIP" in trad.columns else None
    if era is not None and not pd.isna(era) and era > 5.0:
        notes.append(f"High ERA ({era:.2f}) — hittable, aggressive early in counts")
    bb9 = trad.iloc[0].get("BB/9") if not trad.empty and "BB/9" in trad.columns else None
    if bb9 is not None and not pd.isna(bb9) and bb9 > 4.0:
        notes.append(f"Control issues ({bb9:.1f} BB/9) — take pitches early, work counts")
    k9 = trad.iloc[0].get("K/9") if not trad.empty and "K/9" in trad.columns else None
    if k9 is not None and not pd.isna(k9) and k9 < 7.0:
        notes.append(f"Low strikeout stuff ({k9:.1f} K/9) — put the ball in play, don't chase")
    gb_pct = ht.iloc[0].get("Ground%") if not ht.empty and "Ground%" in ht.columns else None
    if gb_pct is not None and not pd.isna(gb_pct) and gb_pct > 50:
        notes.append(f"Ground-ball pitcher ({gb_pct:.1f}% GB) — look to elevate, hit ball in air")
    vel = mov.iloc[0].get("Vel") if not mov.empty and "Vel" in mov.columns else None
    if vel is not None and not pd.isna(vel) and vel < 88:
        notes.append(f"Below-average velocity ({vel:.1f} mph) — sit on offspeed, crush mistakes")
    elif vel is not None and not pd.isna(vel) and vel > 93:
        notes.append(f"High velocity ({vel:.1f} mph) — shorten swing, focus on timing")
    chase_pct = pr.iloc[0].get("Chase%") if not pr.empty and "Chase%" in pr.columns else None
    if chase_pct is not None and not pd.isna(chase_pct) and chase_pct < 22:
        notes.append(f"Low chase induced ({chase_pct:.1f}%) — hitters can be selective")
    if not notes:
        notes.append("Solid pitcher with no glaring weaknesses. Focus on quality at-bats and executing the game plan.")
    return notes


# ══════════════════════════════════════════════════════════════
# GAME PLAN ENGINE — Cross-reference Trackman + TrueMedia
# ══════════════════════════════════════════════════════════════

def _get_opp_hitter_profile(tm, hitter, team):
    """Extract opponent hitter vulnerability profile from TrueMedia."""
    rate = _tm_player(_tm_team(tm["hitting"]["rate"], team), hitter)
    pr = _tm_player(_tm_team(tm["hitting"]["pitch_rates"], team), hitter)
    pt = _tm_player(_tm_team(tm["hitting"]["pitch_types"], team), hitter)
    exit_d = _tm_player(_tm_team(tm["hitting"]["exit"], team), hitter)
    ht = _tm_player(_tm_team(tm["hitting"]["hit_types"], team), hitter)
    hl = _tm_player(_tm_team(tm["hitting"]["hit_locations"], team), hitter)
    pl = _tm_player(_tm_team(tm["hitting"].get("pitch_locations", pd.DataFrame()), team), hitter) if "pitch_locations" in tm["hitting"] else pd.DataFrame()
    sw = _tm_player(_tm_team(tm["hitting"].get("swing_stats", pd.DataFrame()), team), hitter) if "swing_stats" in tm["hitting"] else pd.DataFrame()
    fp = _tm_player(_tm_team(tm["hitting"].get("swing_pct", pd.DataFrame()), team), hitter) if "swing_pct" in tm["hitting"] else pd.DataFrame()
    profile = {
        "name": hitter,
        "bats": rate.iloc[0].get("batsHand", "?") if not rate.empty else "?",
        "pa": _safe_num(rate, "PA"),
        "ops": _safe_num(rate, "OPS"),
        "woba": _safe_num(rate, "WOBA"),
        "k_pct": _safe_num(rate, "K%"),
        "bb_pct": _safe_num(rate, "BB%"),
        "chase_pct": _safe_num(pr, "Chase%"),
        "swstrk_pct": _safe_num(pr, "SwStrk%"),
        "contact_pct": _safe_num(pr, "Contact%"),
        "swing_pct": _safe_num(pr, "Swing%"),
        "iz_swing_pct": _safe_num(sw, "InZoneSwing%"),
        "p_per_pa": _safe_num(pr, "P/PA"),
        "ev": _safe_num(exit_d, "ExitVel"),
        "barrel_pct": _safe_num(exit_d, "Barrel%"),
        "gb_pct": _safe_num(ht, "Ground%"),
        "fb_pct": _safe_num(ht, "Fly%"),
        "ld_pct": _safe_num(ht, "Line%"),
        "pull_pct": _safe_num(hl, "HPull%"),
        # Zone tendencies — where they see pitches
        "high_pct": _safe_num(pl, "High%"),
        "low_pct": _safe_num(pl, "Low%"),
        "inside_pct": _safe_num(pl, "Inside%"),
        "outside_pct": _safe_num(pl, "Outside%"),
        # Platoon splits
        "woba_lhp": _safe_num(sw, "wOBA LHP"),
        "woba_rhp": _safe_num(sw, "wOBA RHP"),
        # 2-strike whiff rates by hand and pitch class
        "whiff_2k_lhp_hard": _safe_num(sw, "2K Whiff vs LHP Hard"),
        "whiff_2k_lhp_os": _safe_num(sw, "2K Whiff vs LHP OS"),
        "whiff_2k_rhp_hard": _safe_num(sw, "2K Whiff vs RHP Hard"),
        "whiff_2k_rhp_os": _safe_num(sw, "2K Whiff vs RHP OS"),
        # First pitch swing rates by pitch type
        "fp_swing_hard_empty": _safe_num(sw, "1PSwing% vs Hard Empty"),
        "fp_swing_ch_empty": _safe_num(sw, "1PSwing% vs CH Empty"),
        "swing_vs_hard": _safe_num(fp, "Swing% vs Hard"),
        "swing_vs_sl": _safe_num(fp, "Swing% vs SL"),
        "swing_vs_cb": _safe_num(fp, "Swing% vs CB"),
        "swing_vs_ch": _safe_num(fp, "Swing% vs CH"),
        "pitch_type_pcts": {},
    }
    if not pt.empty:
        for tm_col, trackman_name in TM_PITCH_PCT_COLS.items():
            v = _safe_num(pt, tm_col)
            if not pd.isna(v) and v > 0:
                profile["pitch_type_pcts"][trackman_name] = v
    return profile


def _get_our_pitcher_arsenal(data, pitcher_name, season_filter=None, tunnel_pop=None):
    """Extract Davidson pitcher arsenal from Trackman data."""
    pdf = filter_davidson(data, role="pitcher")
    pdf = pdf[pdf["Pitcher"] == pitcher_name].copy()
    if season_filter:
        pdf = pdf[pdf["Season"].isin(season_filter)]
    pdf = normalize_pitch_types(pdf)
    pdf = filter_minor_pitches(pdf)
    if len(pdf) < 20:
        return None
    throws = safe_mode(pdf["PitcherThrows"], "Right")
    batter_zones = _build_batter_zones(data)
    _iz = in_zone_mask(pdf, batter_zones, batter_col="Batter")
    _oz = ~_iz & pdf["PlateLocSide"].notna() & pdf["PlateLocHeight"].notna()
    arsenal = {"name": pitcher_name, "throws": throws, "total_pitches": len(pdf), "pitches": {}}
    stuff_df = _compute_stuff_plus(pdf)
    for pt_name, grp in pdf.groupby("TaggedPitchType"):
        if len(grp) < 5:
            continue
        sw = grp[grp["PitchCall"].isin(SWING_CALLS)]
        wh = grp[grp["PitchCall"] == "StrikeSwinging"]
        csw = grp[grp["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"])]
        ip = grp[grp["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
        oz_grp = grp[_oz.reindex(grp.index, fill_value=False)]
        oz_sw = oz_grp[oz_grp["PitchCall"].isin(SWING_CALLS)]
        iz_grp = grp[_iz.reindex(grp.index, fill_value=False)]
        stuff_plus = np.nan
        if not stuff_df.empty and "StuffPlus" in stuff_df.columns:
            sp = stuff_df[stuff_df["TaggedPitchType"] == pt_name]["StuffPlus"]
            if len(sp) > 0:
                stuff_plus = sp.mean()
        # Per-pitch zone effectiveness (where this pitch is best from Trackman)
        loc_grp = grp.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        zone_mid_h = (ZONE_HEIGHT_BOT + ZONE_HEIGHT_TOP) / 2
        zone_eff = {}
        for zn, zmask in [
            ("up", loc_grp["PlateLocHeight"] >= zone_mid_h),
            ("down", (loc_grp["PlateLocHeight"] < zone_mid_h) & (loc_grp["PlateLocHeight"] >= ZONE_HEIGHT_BOT)),
            ("glove", loc_grp["PlateLocSide"] > 0 if throws == "Right" else loc_grp["PlateLocSide"] < 0),
            ("arm", loc_grp["PlateLocSide"] <= 0 if throws == "Right" else loc_grp["PlateLocSide"] >= 0),
            ("chase_low", loc_grp["PlateLocHeight"] < ZONE_HEIGHT_BOT),
        ]:
            zdf = loc_grp[zmask]
            if len(zdf) >= 5:
                z_sw = zdf[zdf["PitchCall"].isin(SWING_CALLS)]
                z_wh = zdf[zdf["PitchCall"] == "StrikeSwinging"]
                z_ip = zdf[zdf["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
                zone_eff[zn] = {
                    "n": len(zdf), "whiff_pct": len(z_wh) / max(len(z_sw), 1) * 100 if len(z_sw) > 0 else np.nan,
                    "csw_pct": len(zdf[zdf["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"])]) / len(zdf) * 100,
                    "ev_against": z_ip["ExitSpeed"].mean() if len(z_ip) > 0 else np.nan,
                }
        # Compute effective velocity estimate
        eff_velo = np.nan
        if loc_grp["RelSpeed"].notna().any() and len(loc_grp) >= 5:
            loc_adj = (loc_grp["PlateLocHeight"] - 2.5) * 1.5 + loc_grp["PlateLocSide"].abs() * (-0.5)
            eff_velo = (loc_grp["RelSpeed"] + loc_adj).mean()
        # Barrel% against
        barrels_against = int(is_barrel_mask(ip).sum()) if len(ip) > 0 else 0
        barrel_pct_against = barrels_against / max(len(ip), 1) * 100 if len(ip) > 0 else np.nan
        # Extension
        ext_val = grp["Extension"].mean() if "Extension" in grp.columns and grp["Extension"].notna().any() else np.nan
        arsenal["pitches"][pt_name] = {
            "usage_pct": len(grp) / len(pdf) * 100,
            "avg_velo": grp["RelSpeed"].mean() if grp["RelSpeed"].notna().any() else np.nan,
            "avg_spin": grp["SpinRate"].mean() if grp["SpinRate"].notna().any() else np.nan,
            "ivb": grp["InducedVertBreak"].mean() if "InducedVertBreak" in grp.columns and grp["InducedVertBreak"].notna().any() else np.nan,
            "hb": grp["HorzBreak"].mean() if "HorzBreak" in grp.columns and grp["HorzBreak"].notna().any() else np.nan,
            "whiff_pct": len(wh) / max(len(sw), 1) * 100 if len(sw) > 0 else np.nan,
            "csw_pct": len(csw) / len(grp) * 100,
            "chase_pct": len(oz_sw) / max(len(oz_grp), 1) * 100 if len(oz_grp) > 0 else np.nan,
            "ev_against": ip["ExitSpeed"].mean() if len(ip) > 0 else np.nan,
            "barrel_pct_against": barrel_pct_against,
            "stuff_plus": stuff_plus,
            "count": len(grp),
            "eff_velo": eff_velo,
            "extension": ext_val,
            "zone_eff": zone_eff,
        }
    # Tunnel scores and pitch pair sequencing results
    arsenal["tunnels"] = _compute_tunnel_score(pdf, tunnel_pop=tunnel_pop)
    arsenal["sequences"] = _compute_pitch_pair_results(pdf, data)
    return arsenal


def _get_opp_pitcher_profile(tm, pitcher_name, team):
    """Extract opponent pitcher profile from TrueMedia — full enrichment."""
    trad = _tm_player(_tm_team(tm["pitching"]["traditional"], team), pitcher_name)
    rate = _tm_player(_tm_team(tm["pitching"]["rate"], team), pitcher_name)
    mov = _tm_player(_tm_team(tm["pitching"]["movement"], team), pitcher_name)
    pt = _tm_player(_tm_team(tm["pitching"]["pitch_types"], team), pitcher_name)
    pr = _tm_player(_tm_team(tm["pitching"]["pitch_rates"], team), pitcher_name)
    exit_d = _tm_player(_tm_team(tm["pitching"]["exit"], team), pitcher_name)
    ht = _tm_player(_tm_team(tm["pitching"]["hit_types"], team), pitcher_name)
    pl = _tm_player(_tm_team(tm["pitching"].get("pitch_locations", pd.DataFrame()), team), pitcher_name)
    xr = _tm_player(_tm_team(tm["pitching"].get("expected_rate", pd.DataFrame()), team), pitcher_name)
    pc = _tm_player(_tm_team(tm["pitching"].get("pitch_counts", pd.DataFrame()), team), pitcher_name)
    profile = {
        "name": pitcher_name,
        "throws": trad.iloc[0].get("throwsHand", "?") if not trad.empty else "?",
        # Traditional
        "era": _safe_num(trad, "ERA"), "fip": _safe_num(trad, "FIP"),
        "ip": _safe_num(trad, "IP"), "gs": _safe_num(trad, "GS"),
        "k9": _safe_num(trad, "K/9"), "bb9": _safe_num(trad, "BB/9"),
        "whip": _safe_num(trad, "WHIP"), "hr9": _safe_num(trad, "HR/9"),
        # Rate
        "xfip": _safe_num(rate, "xFIP"), "woba": _safe_num(rate, "wOBA"),
        "lob_pct": _safe_num(rate, "LOB%"), "k_pct": _safe_num(rate, "K%"),
        "bb_pct": _safe_num(rate, "BB%"), "ops_against": _safe_num(rate, "OPS"),
        # Movement
        "velo": _safe_num(mov, "Vel"), "max_velo": _safe_num(mov, "MxVel"),
        "velo_range": _safe_num(mov, "VelRange"), "spin": _safe_num(mov, "Spin"),
        "ivb": _safe_num(mov, "IndVertBrk"), "hb": _safe_num(mov, "HorzBrk"),
        "extension": _safe_num(mov, "Extension"), "eff_velo": _safe_num(mov, "EffectVel"),
        "vaa": _safe_num(mov, "VertApprAngle"),
        # Pitch rates / command
        "chase_pct": _safe_num(pr, "Chase%"), "swstrk_pct": _safe_num(pr, "SwStrk%"),
        "contact_pct": _safe_num(pr, "Contact%"), "miss_pct": _safe_num(pr, "Miss%"),
        "inzone_pct": _safe_num(pr, "InZone%"), "comploc_pct": _safe_num(pr, "CompLoc%"),
        "swing_pct": _safe_num(pr, "Swing%"), "callstrk_pct": _safe_num(pr, "CallStrk%"),
        "p_per_bf": _safe_num(pr, "P/BF"),
        # Pitch locations
        "loc_high_pct": _safe_num(pl, "High%"), "loc_vmid_pct": _safe_num(pl, "VMid%"),
        "loc_low_pct": _safe_num(pl, "Low%"), "loc_inside_pct": _safe_num(pl, "Inside%"),
        "loc_hmid_pct": _safe_num(pl, "HMid%"), "loc_outside_pct": _safe_num(pl, "Outside%"),
        "loc_uphalf_pct": _safe_num(pl, "UpHalf%"), "loc_lowhalf_pct": _safe_num(pl, "LowHalf%"),
        "loc_inhalf_pct": _safe_num(pl, "InHalf%"), "loc_outhalf_pct": _safe_num(pl, "OutHalf%"),
        # Exit data
        "ev_against": _safe_num(exit_d, "ExitVel"), "barrel_pct": _safe_num(exit_d, "Barrel%"),
        "hard_hit_pct": _safe_num(exit_d, "HardOut"), "launch_ang": _safe_num(exit_d, "LaunchAng"),
        # Hit types
        "gb_pct": _safe_num(ht, "Ground%"), "fb_pct": _safe_num(ht, "Fly%"),
        "ld_pct": _safe_num(ht, "Line%"),
        # Expected
        "xavg": _safe_num(xr, "xAVG"), "xslg": _safe_num(xr, "xSLG"),
        "xwoba": _safe_num(xr, "xWOBA"),
        # Arsenal
        "pitch_mix": {},
        # Pitch counts (raw numbers for sample size)
        "total_pitches": _safe_num(pc, "P"),
    }
    # Build pitch mix
    if not pt.empty:
        for tm_col, trackman_name in TM_PITCH_PCT_COLS.items():
            v = _safe_num(pt, tm_col)
            if not pd.isna(v) and v > 0:
                profile["pitch_mix"][trackman_name] = v
    # Derive: primary pitch (highest usage)
    if profile["pitch_mix"]:
        profile["primary_pitch"] = max(profile["pitch_mix"], key=profile["pitch_mix"].get)
        # Putaway candidates: offspeed/breaking pitches (non-hard)
        _hard = {"Fastball", "Sinker", "Cutter"}
        profile["putaway_candidates"] = {p: u for p, u in profile["pitch_mix"].items() if p not in _hard and u >= 5}
        # Location tendencies summary
        h = profile["loc_high_pct"]
        l = profile["loc_low_pct"]
        i = profile["loc_inside_pct"]
        o = profile["loc_outside_pct"]
        parts = []
        if not pd.isna(h) and h >= 30:
            parts.append(f"high ({h:.0f}%)")
        if not pd.isna(l) and l >= 35:
            parts.append(f"low ({l:.0f}%)")
        if not pd.isna(i) and i >= 30:
            parts.append(f"inside ({i:.0f}%)")
        if not pd.isna(o) and o >= 30:
            parts.append(f"outside ({o:.0f}%)")
        profile["location_tendency"] = ", ".join(parts) if parts else "balanced"
    else:
        profile["primary_pitch"] = None
        profile["putaway_candidates"] = {}
        profile["location_tendency"] = "unknown"
    return profile


def _get_our_hitter_profile(data, batter_name, season_filter=None):
    """Extract Davidson hitter profile from Trackman data — enriched with zones, counts, pitch-class."""
    bdf = filter_davidson(data, role="batter")
    bdf = bdf[bdf["Batter"] == batter_name].copy()
    if season_filter:
        bdf = bdf[bdf["Season"].isin(season_filter)]
    bdf = normalize_pitch_types(bdf)
    if len(bdf) < 20:
        return None
    bats = safe_mode(bdf["BatterSide"], "Right")
    batter_zones = _build_batter_zones(data)
    _iz = in_zone_mask(bdf, batter_zones, batter_col="Batter")
    _oz = ~_iz & bdf["PlateLocSide"].notna() & bdf["PlateLocHeight"].notna()
    profile = {"name": batter_name, "bats": bats, "total_pitches": len(bdf), "by_pitch_type": {}}
    for pt_name, grp in bdf.groupby("TaggedPitchType"):
        if len(grp) < 10:
            continue
        sw = grp[grp["PitchCall"].isin(SWING_CALLS)]
        wh = grp[grp["PitchCall"] == "StrikeSwinging"]
        ip = grp[grp["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
        barrels = int(is_barrel_mask(ip).sum()) if len(ip) > 0 else 0
        pt_entry = {
            "seen": len(grp),
            "swing_pct": len(sw) / len(grp) * 100,
            "whiff_pct": len(wh) / max(len(sw), 1) * 100 if len(sw) > 0 else np.nan,
            "avg_ev": ip["ExitSpeed"].mean() if len(ip) > 0 else np.nan,
            "barrel_pct": barrels / max(len(ip), 1) * 100 if len(ip) > 0 else np.nan,
            "hard_hit_pct": (len(ip[ip["ExitSpeed"] >= 95]) / max(len(ip), 1) * 100) if len(ip) > 0 else np.nan,
        }
        # Per-pitch contact depth
        if "EffectiveVelo" in grp.columns and "RelSpeed" in grp.columns:
            cd_df = ip.dropna(subset=["EffectiveVelo", "RelSpeed"])
            if len(cd_df) > 0:
                pt_entry["contact_depth"] = (cd_df["EffectiveVelo"] - cd_df["RelSpeed"]).mean()
            else:
                pt_entry["contact_depth"] = np.nan
        else:
            pt_entry["contact_depth"] = np.nan
        # Per-pitch bat speed proxy
        if "RelSpeed" in ip.columns and len(ip) > 0:
            bs_df = ip.dropna(subset=["RelSpeed"])
            if len(bs_df) > 0:
                pt_entry["bat_speed"] = ((bs_df["ExitSpeed"] - 0.2 * bs_df["RelSpeed"]) / 1.2).mean()
            else:
                pt_entry["bat_speed"] = np.nan
        else:
            pt_entry["bat_speed"] = np.nan
        # Hard-hit launch angle
        ip_la = ip.dropna(subset=["Angle"])
        hh_ip = ip_la[ip_la["ExitSpeed"] >= 95] if len(ip_la) > 0 else ip_la
        pt_entry["hard_hit_la"] = hh_ip["Angle"].median() if len(hh_ip) >= 3 else np.nan
        profile["by_pitch_type"][pt_name] = pt_entry

    all_sw = bdf[bdf["PitchCall"].isin(SWING_CALLS)]
    all_ip = bdf[bdf["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
    oz_pitches = bdf[_oz.reindex(bdf.index, fill_value=False)]
    oz_sw = oz_pitches[oz_pitches["PitchCall"].isin(SWING_CALLS)]
    # overall stats: EV, barrel rate, sweet spot%
    all_barrels = int(is_barrel_mask(all_ip).sum()) if len(all_ip) > 0 else 0
    all_ip_la = all_ip.dropna(subset=["Angle"]) if "Angle" in all_ip.columns else pd.DataFrame()
    sweet_spot_n = int(((all_ip_la["Angle"] >= 8) & (all_ip_la["Angle"] <= 32)).sum()) if len(all_ip_la) > 0 else 0
    profile["overall"] = {
        "avg_ev": all_ip["ExitSpeed"].mean() if len(all_ip) > 0 else np.nan,
        "barrel_pct": all_barrels / max(len(all_ip), 1) * 100 if len(all_ip) > 0 else np.nan,
        "barrel_count": all_barrels,
        "sweet_spot_pct": sweet_spot_n / max(len(all_ip_la), 1) * 100 if len(all_ip_la) > 0 else np.nan,
        "hard_hit_pct": (all_ip["ExitSpeed"] >= 87).sum() / max(len(all_ip), 1) * 100 if len(all_ip) > 0 else np.nan,
    }

    # ── Zone quadrant performance ──
    has_loc = bdf["PlateLocSide"].notna() & bdf["PlateLocHeight"].notna()
    loc_df = bdf[has_loc].copy()
    zone_mid_height = (ZONE_HEIGHT_BOT + ZONE_HEIGHT_TOP) / 2  # ~2.5 ft
    zone_quads = {
        "up_in":    loc_df[(loc_df["PlateLocHeight"] >= zone_mid_height) & (loc_df["PlateLocSide"] <= 0)],
        "up_away":  loc_df[(loc_df["PlateLocHeight"] >= zone_mid_height) & (loc_df["PlateLocSide"] > 0)],
        "down_in":  loc_df[(loc_df["PlateLocHeight"] < zone_mid_height) & (loc_df["PlateLocHeight"] >= ZONE_HEIGHT_BOT) & (loc_df["PlateLocSide"] <= 0)],
        "down_away":loc_df[(loc_df["PlateLocHeight"] < zone_mid_height) & (loc_df["PlateLocHeight"] >= ZONE_HEIGHT_BOT) & (loc_df["PlateLocSide"] > 0)],
        "heart":    loc_df[(loc_df["PlateLocHeight"].between(zone_mid_height - 0.4, zone_mid_height + 0.4)) & (loc_df["PlateLocSide"].abs() <= 0.4)],
        "chase_up": loc_df[loc_df["PlateLocHeight"] > ZONE_HEIGHT_TOP],
        "chase_down":loc_df[loc_df["PlateLocHeight"] < ZONE_HEIGHT_BOT],
        "chase_in": loc_df[(loc_df["PlateLocSide"] < -ZONE_SIDE) & loc_df["PlateLocHeight"].between(ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP)],
        "chase_away":loc_df[(loc_df["PlateLocSide"] > ZONE_SIDE) & loc_df["PlateLocHeight"].between(ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP)],
    }
    profile["zones"] = {}
    for zname, zdf in zone_quads.items():
        if len(zdf) < 5:
            profile["zones"][zname] = {"n": len(zdf), "swing_pct": np.nan, "whiff_pct": np.nan, "avg_ev": np.nan}
            continue
        z_sw = zdf[zdf["PitchCall"].isin(SWING_CALLS)]
        z_wh = zdf[zdf["PitchCall"] == "StrikeSwinging"]
        z_ip = zdf[zdf["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
        profile["zones"][zname] = {
            "n": len(zdf),
            "swing_pct": len(z_sw) / len(zdf) * 100,
            "whiff_pct": len(z_wh) / max(len(z_sw), 1) * 100 if len(z_sw) > 0 else np.nan,
            "avg_ev": z_ip["ExitSpeed"].mean() if len(z_ip) > 0 else np.nan,
        }

    # ── Count EV Grid (per-count performance) ──
    profile["count_ev_grid"] = {}
    if "Balls" in bdf.columns and "Strikes" in bdf.columns:
        cdf = bdf.dropna(subset=["Balls", "Strikes"]).copy()
        cdf["_b"] = cdf["Balls"].astype(int)
        cdf["_s"] = cdf["Strikes"].astype(int)
        for b_val in range(4):
            for s_val in range(3):
                mask = (cdf["_b"] == b_val) & (cdf["_s"] == s_val)
                g = cdf[mask]
                if len(g) < 5:
                    continue
                c_sw = g[g["PitchCall"].isin(SWING_CALLS)]
                c_ip = g[g["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
                count_key = f"{b_val}-{s_val}"
                profile["count_ev_grid"][count_key] = {
                    "n": len(g),
                    "swing_pct": len(c_sw) / len(g) * 100,
                    "avg_ev": c_ip["ExitSpeed"].mean() if len(c_ip) > 0 else np.nan,
                }

    # ── Count group performance (kept for backwards compat) ──
    _count_groups = {
        "early": [(0, 0), (1, 0), (0, 1)],
        "ahead": [(2, 0), (3, 0), (3, 1), (2, 1)],
        "behind": [(0, 2), (1, 2)],
        "even": [(1, 1), (2, 2)],
        "full": [(3, 2)],
    }
    profile["by_count"] = {}
    if "Balls" in bdf.columns and "Strikes" in bdf.columns:
        cdf = bdf.dropna(subset=["Balls", "Strikes"]).copy()
        cdf["_b"] = cdf["Balls"].astype(int)
        cdf["_s"] = cdf["Strikes"].astype(int)
        for cg_name, counts in _count_groups.items():
            mask = pd.Series(False, index=cdf.index)
            for b, s in counts:
                mask |= (cdf["_b"] == b) & (cdf["_s"] == s)
            g = cdf[mask]
            if len(g) < 5:
                profile["by_count"][cg_name] = {"n": len(g), "swing_pct": np.nan, "whiff_pct": np.nan, "avg_ev": np.nan}
                continue
            c_sw = g[g["PitchCall"].isin(SWING_CALLS)]
            c_wh = g[g["PitchCall"] == "StrikeSwinging"]
            c_ip = g[g["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            profile["by_count"][cg_name] = {
                "n": len(g),
                "swing_pct": len(c_sw) / len(g) * 100,
                "whiff_pct": len(c_wh) / max(len(c_sw), 1) * 100 if len(c_sw) > 0 else np.nan,
                "avg_ev": c_ip["ExitSpeed"].mean() if len(c_ip) > 0 else np.nan,
            }

    # ── Pitch-class performance (Hard vs Offspeed) ──
    _hard_types = {"Fastball", "Sinker", "Cutter"}
    _os_types = {"Slider", "Curveball", "Changeup", "Splitter", "Sweeper", "Knuckle Curve"}
    profile["by_pitch_class"] = {}
    for cls_name, cls_types in [("hard", _hard_types), ("offspeed", _os_types)]:
        g = bdf[bdf["TaggedPitchType"].isin(cls_types)]
        if len(g) < 5:
            profile["by_pitch_class"][cls_name] = {"n": len(g), "whiff_pct": np.nan, "avg_ev": np.nan, "chase_pct": np.nan}
            continue
        cls_sw = g[g["PitchCall"].isin(SWING_CALLS)]
        cls_wh = g[g["PitchCall"] == "StrikeSwinging"]
        cls_ip = g[g["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
        cls_oz = g[g.index.isin(oz_pitches.index)]
        cls_oz_sw = cls_oz[cls_oz["PitchCall"].isin(SWING_CALLS)]
        profile["by_pitch_class"][cls_name] = {
            "n": len(g),
            "whiff_pct": len(cls_wh) / max(len(cls_sw), 1) * 100 if len(cls_sw) > 0 else np.nan,
            "avg_ev": cls_ip["ExitSpeed"].mean() if len(cls_ip) > 0 else np.nan,
            "chase_pct": len(cls_oz_sw) / max(len(cls_oz), 1) * 100 if len(cls_oz) > 0 else np.nan,
        }

    # ── Swing Path Metrics (from hit lab) ──
    inplay_full = bdf[bdf["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
    inplay_la = inplay_full.dropna(subset=["Angle"])
    sp = {}
    if len(inplay_la) >= 10:
        ev_75 = inplay_la["ExitSpeed"].quantile(0.75)
        hard_hit = inplay_la[inplay_la["ExitSpeed"] >= ev_75]
        if len(hard_hit) >= 5:
            sp["attack_angle"] = hard_hit["Angle"].median()
            sp["avg_la_all"] = inplay_la["Angle"].median()
            # Swing type classification
            aa = sp["attack_angle"]
            if aa > 20:
                sp["swing_type"] = "Steep Uppercut"
            elif aa > 14:
                sp["swing_type"] = "Lift-Oriented"
            elif aa > 8:
                sp["swing_type"] = "Slight Uppercut"
            elif aa > 2:
                sp["swing_type"] = "Level"
            else:
                sp["swing_type"] = "Downward/Chopper"
            # Bat speed proxy — empirical: EV ≈ 0.2*PS + 1.2*BS
            if "RelSpeed" in hard_hit.columns:
                hh_sp = hard_hit.dropna(subset=["RelSpeed"])
                if len(hh_sp) > 0:
                    bs = (hh_sp["ExitSpeed"] - 0.2 * hh_sp["RelSpeed"]) / 1.2
                    sp["bat_speed_avg"] = bs.mean()
                    sp["bat_speed_max"] = bs.max()
            # Contact depth
            if "EffectiveVelo" in hard_hit.columns and "RelSpeed" in hard_hit.columns:
                cd_df = hard_hit.dropna(subset=["EffectiveVelo", "RelSpeed"])
                if len(cd_df) > 0:
                    depth_val = (cd_df["EffectiveVelo"] - cd_df["RelSpeed"]).mean()
                    sp["contact_depth"] = depth_val
                    sp["depth_label"] = "Out Front" if depth_val > 1.5 else ("Deep Contact" if depth_val < -1.5 else "Neutral")
                else:
                    sp["contact_depth"] = np.nan
                    sp["depth_label"] = "Unknown"
            # Path adjust: how LA changes with pitch height
            hh_loc = hard_hit.dropna(subset=["PlateLocHeight"])
            if len(hh_loc) >= 8:
                from scipy import stats as sp_stats
                slope, _, _, _, _ = sp_stats.linregress(hh_loc["PlateLocHeight"], hh_loc["Angle"])
                sp["path_adjust"] = slope  # degrees per foot
            # Per-pitch-type swing path
            sp_by_pt = {}
            for pt_name2, ptg in hard_hit.groupby("TaggedPitchType"):
                if len(ptg) < 3:
                    continue
                pt_ip = ptg.dropna(subset=["ExitSpeed"])
                sp_by_pt[pt_name2] = {
                    "hard_hit_la": ptg["Angle"].median(),
                    "hard_hit_ev": pt_ip["ExitSpeed"].mean() if len(pt_ip) > 0 else np.nan,
                }
                if "RelSpeed" in ptg.columns:
                    ptg_sp = ptg.dropna(subset=["RelSpeed"])
                    if len(ptg_sp) > 0:
                        sp_by_pt[pt_name2]["bat_speed"] = ((ptg_sp["ExitSpeed"] - 0.2 * ptg_sp["RelSpeed"]) / 1.2).mean()
            sp["by_pitch_type"] = sp_by_pt
    profile["swing_path"] = sp if sp else None

    # ── Discipline Metrics ──
    iz_mask = _iz.reindex(bdf.index, fill_value=False)
    oz_mask = _oz.reindex(bdf.index, fill_value=False)
    all_swings = bdf[bdf["PitchCall"].isin(SWING_CALLS)]
    all_whiffs = bdf[bdf["PitchCall"] == "StrikeSwinging"]
    iz_pitches = bdf[iz_mask]
    iz_swings = iz_pitches[iz_pitches["PitchCall"].isin(SWING_CALLS)]
    iz_contacts = iz_pitches[iz_pitches["PitchCall"].isin(CONTACT_CALLS)]
    oz_pitches_d = bdf[oz_mask]
    oz_swings_d = oz_pitches_d[oz_pitches_d["PitchCall"].isin(SWING_CALLS)]
    # PA-based stats — count unique PAs using composite PA identifier
    _pa_cols = ["GameID", "Inning", "PAofInning", "Batter"]
    if all(c in bdf.columns for c in _pa_cols):
        pa_count = bdf.drop_duplicates(subset=_pa_cols).shape[0]
        ks = bdf[bdf["KorBB"] == "Strikeout"].drop_duplicates(subset=_pa_cols).shape[0] if "KorBB" in bdf.columns else 0
        bbs = bdf[bdf["KorBB"] == "Walk"].drop_duplicates(subset=_pa_cols).shape[0] if "KorBB" in bdf.columns else 0
    else:
        pa_count = bdf["PitchofPA"].eq(1).sum() if "PitchofPA" in bdf.columns else np.nan
        ks = len(bdf[bdf["KorBB"] == "Strikeout"]) if "KorBB" in bdf.columns else 0
        bbs = len(bdf[bdf["KorBB"] == "Walk"]) if "KorBB" in bdf.columns else 0
    profile["discipline"] = {
        "chase_pct": len(oz_swings_d) / max(len(oz_pitches_d), 1) * 100 if len(oz_pitches_d) > 0 else np.nan,
        "whiff_pct": len(all_whiffs) / max(len(all_swings), 1) * 100 if len(all_swings) > 0 else np.nan,
        "swing_pct": len(all_swings) / max(len(bdf), 1) * 100,
        "z_swing_pct": len(iz_swings) / max(len(iz_pitches), 1) * 100 if len(iz_pitches) > 0 else np.nan,
        "z_contact_pct": len(iz_contacts) / max(len(iz_swings), 1) * 100 if len(iz_swings) > 0 else np.nan,
        "k_pct": ks / max(pa_count, 1) * 100 if not pd.isna(pa_count) and pa_count > 0 else np.nan,
        "bb_pct": bbs / max(pa_count, 1) * 100 if not pd.isna(pa_count) and pa_count > 0 else np.nan,
    }

    # ── First-Pitch Approach ──
    if "PitchofPA" in bdf.columns:
        fp = bdf[bdf["PitchofPA"] == 1]
        if len(fp) >= 5:
            fp_sw = fp[fp["PitchCall"].isin(SWING_CALLS)]
            fp_wh = fp[fp["PitchCall"] == "StrikeSwinging"]
            fp_ip = fp[fp["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            profile["first_pitch"] = {
                "n": len(fp),
                "swing_pct": len(fp_sw) / len(fp) * 100,
                "whiff_pct": len(fp_wh) / max(len(fp_sw), 1) * 100 if len(fp_sw) > 0 else np.nan,
                "avg_ev": fp_ip["ExitSpeed"].mean() if len(fp_ip) > 0 else np.nan,
            }
        else:
            profile["first_pitch"] = None
    else:
        profile["first_pitch"] = None

    # ── 2-Strike Adjustments ──
    if "Strikes" in bdf.columns:
        two_k = bdf[bdf["Strikes"].astype(float) >= 2]
        pre_2k = bdf[bdf["Strikes"].astype(float) < 2]
        if len(two_k) >= 10:
            tk_sw = two_k[two_k["PitchCall"].isin(SWING_CALLS)]
            tk_wh = two_k[two_k["PitchCall"] == "StrikeSwinging"]
            tk_ip = two_k[two_k["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            pre_sw = pre_2k[pre_2k["PitchCall"].isin(SWING_CALLS)]
            pre_wh = pre_2k[pre_2k["PitchCall"] == "StrikeSwinging"]
            profile["two_strike"] = {
                "n": len(two_k),
                "swing_pct": len(tk_sw) / max(len(two_k), 1) * 100,
                "whiff_pct": len(tk_wh) / max(len(tk_sw), 1) * 100 if len(tk_sw) > 0 else np.nan,
                "avg_ev": tk_ip["ExitSpeed"].mean() if len(tk_ip) > 0 else np.nan,
                "pre_2k_whiff": len(pre_wh) / max(len(pre_sw), 1) * 100 if len(pre_sw) > 0 else np.nan,
            }
        else:
            profile["two_strike"] = None
    else:
        profile["two_strike"] = None

    # ── Zone Coverage Grid (5x5) ──
    zone_grid = {}
    if len(loc_df) >= 20:
        bside = {"Right": "Right", "Left": "Left"}.get(bats, "Right")
        h_edges = [-2, -0.83, -0.28, 0.28, 0.83, 2]
        v_edges = [0.5, 1.5, 2.17, 2.83, 3.5, 4.5]
        col_labels = ["Far In", "Inside", "Middle", "Outside", "Far Out"]
        row_labels = ["Low+", "Low", "Mid", "High", "High+"]
        if bside == "Left":
            col_labels = col_labels[::-1]
        for ri in range(5):
            for ci in range(5):
                cell_mask = (
                    (loc_df["PlateLocSide"] >= h_edges[ci]) &
                    (loc_df["PlateLocSide"] < h_edges[ci + 1]) &
                    (loc_df["PlateLocHeight"] >= v_edges[ri]) &
                    (loc_df["PlateLocHeight"] < v_edges[ri + 1])
                )
                cell = loc_df[cell_mask]
                if len(cell) < 5:
                    continue
                cell_sw = cell[cell["PitchCall"].isin(SWING_CALLS)]
                cell_wh = cell[cell["PitchCall"] == "StrikeSwinging"]
                cell_con = cell[cell["PitchCall"].isin(CONTACT_CALLS)]
                cell_ip = cell[cell["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
                key = f"{row_labels[ri]}_{col_labels[ci]}"
                zone_grid[key] = {
                    "n": len(cell),
                    "swing_pct": len(cell_sw) / len(cell) * 100,
                    "whiff_pct": len(cell_wh) / max(len(cell_sw), 1) * 100 if len(cell_sw) > 0 else np.nan,
                    "contact_rate": len(cell_con) / max(len(cell_sw), 1) * 100 if len(cell_sw) > 0 else np.nan,
                    "avg_ev": cell_ip["ExitSpeed"].mean() if len(cell_ip) > 0 else np.nan,
                }
    profile["zone_grid"] = zone_grid

    # ── 1A. Per-pitch-type zone_damage (6-zone map) ──
    zone_mid = (ZONE_HEIGHT_BOT + ZONE_HEIGHT_TOP) / 2  # ~2.5 ft
    for pt_name, pt_entry in profile["by_pitch_type"].items():
        pt_grp = loc_df[loc_df["TaggedPitchType"] == pt_name]
        pt_ip = pt_grp[pt_grp["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
        if len(pt_ip) < 10:
            pt_entry["zone_damage"] = {}
            continue
        # 6-zone: up/mid/down × in/out (relative to batter hand)
        is_rhh = bats == "Right"
        zd = {}
        for v_label, v_lo, v_hi in [("up", zone_mid + 0.5, 5.0),
                                      ("mid", zone_mid - 0.5, zone_mid + 0.5),
                                      ("down", 0.0, zone_mid - 0.5)]:
            for h_label, h_cond in [("in", pt_ip["PlateLocSide"] < 0 if is_rhh else pt_ip["PlateLocSide"] > 0),
                                     ("out", pt_ip["PlateLocSide"] >= 0 if is_rhh else pt_ip["PlateLocSide"] <= 0)]:
                cell = pt_ip[(pt_ip["PlateLocHeight"] >= v_lo) &
                             (pt_ip["PlateLocHeight"] < v_hi) & h_cond]
                if len(cell) < 5:
                    continue
                cell_barrels = int(is_barrel_mask(cell).sum()) if len(cell) > 0 else 0
                zd[f"{v_label}_{h_label}"] = {
                    "n": len(cell),
                    "avg_ev": cell["ExitSpeed"].mean(),
                    "barrels": cell_barrels,
                }
        pt_entry["zone_damage"] = zd

    # ── 1B. 2-strike whiff by pitch class (hard vs offspeed) ──
    if profile.get("two_strike") and "Strikes" in bdf.columns:
        two_k = bdf[bdf["Strikes"].astype(float) >= 2]
        if len(two_k) >= 10:
            _hard_cls = {"Fastball", "Sinker", "Cutter"}
            _os_cls = {"Slider", "Curveball", "Changeup", "Splitter", "Sweeper", "Knuckle Curve"}
            for cls_label, cls_set in [("whiff_hard", _hard_cls), ("whiff_os", _os_cls)]:
                cls_2k = two_k[two_k["TaggedPitchType"].isin(cls_set)]
                cls_2k_sw = cls_2k[cls_2k["PitchCall"].isin(SWING_CALLS)]
                cls_2k_wh = cls_2k[cls_2k["PitchCall"] == "StrikeSwinging"]
                if len(cls_2k_sw) >= 8:
                    profile["two_strike"][cls_label] = len(cls_2k_wh) / len(cls_2k_sw) * 100

    # ── 1C. Barrel zone concentration (top 3 cells by barrel count) ──
    barrel_zones = []
    if len(loc_df) >= 20 and zone_grid:
        all_ip_loc = loc_df[loc_df["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed", "Angle"])
        if len(all_ip_loc) >= 10:
            bside_1c = {"Right": "Right", "Left": "Left"}.get(bats, "Right")
            h_edges_1c = [-2, -0.83, -0.28, 0.28, 0.83, 2]
            v_edges_1c = [0.5, 1.5, 2.17, 2.83, 3.5, 4.5]
            col_labels_1c = ["Far In", "Inside", "Middle", "Outside", "Far Out"]
            row_labels_1c = ["Low+", "Low", "Mid", "High", "High+"]
            if bside_1c == "Left":
                col_labels_1c = col_labels_1c[::-1]
            cell_barrels_list = []
            for ri in range(5):
                for ci in range(5):
                    cm = ((all_ip_loc["PlateLocSide"] >= h_edges_1c[ci]) &
                          (all_ip_loc["PlateLocSide"] < h_edges_1c[ci + 1]) &
                          (all_ip_loc["PlateLocHeight"] >= v_edges_1c[ri]) &
                          (all_ip_loc["PlateLocHeight"] < v_edges_1c[ri + 1]))
                    cell_ip = all_ip_loc[cm]
                    if len(cell_ip) < 8:
                        continue
                    b_count = int(is_barrel_mask(cell_ip).sum())
                    if b_count >= 2:
                        ev_avg = cell_ip["ExitSpeed"].mean()
                        key_1c = f"{row_labels_1c[ri]}_{col_labels_1c[ci]}"
                        cell_barrels_list.append((key_1c, b_count, round(ev_avg, 1)))
            cell_barrels_list.sort(key=lambda x: x[1], reverse=True)
            barrel_zones = cell_barrels_list[:3]
    profile["barrel_zones"] = barrel_zones

    # ── 1D. Spray direction (pull/center/oppo) ──
    spray = {}
    if "Direction" in bdf.columns:
        dir_ip = bdf[(bdf["PitchCall"] == "InPlay") & bdf["Direction"].notna()].copy()
        if len(dir_ip) >= 20:
            is_rhh_spray = bats == "Right"
            if is_rhh_spray:
                pull_mask = dir_ip["Direction"] < -15
                oppo_mask = dir_ip["Direction"] > 15
            else:
                pull_mask = dir_ip["Direction"] > 15
                oppo_mask = dir_ip["Direction"] < -15
            center_mask = ~pull_mask & ~oppo_mask
            total = len(dir_ip)
            spray = {
                "pull_pct": round(pull_mask.sum() / total * 100, 1),
                "center_pct": round(center_mask.sum() / total * 100, 1),
                "oppo_pct": round(oppo_mask.sum() / total * 100, 1),
            }
    profile["spray"] = spray

    return profile


# ── Scoring Engine ──

_hard_pitches = {"Fastball", "Sinker", "Cutter"}
_swing_map = {
    "Fastball": "swing_vs_hard", "Sinker": "swing_vs_hard", "Cutter": "swing_vs_hard",
    "Slider": "swing_vs_sl", "Sweeper": "swing_vs_sl",
    "Curveball": "swing_vs_cb", "Knuckle Curve": "swing_vs_cb",
    "Changeup": "swing_vs_ch", "Splitter": "swing_vs_ch",
}


def _lookup_tunnel(a, b, tun_df):
    """Lookup tunnel score and grade for a pitch pair."""
    if not isinstance(tun_df, pd.DataFrame) or tun_df.empty:
        return np.nan, "-"
    m = tun_df[((tun_df["Pitch A"]==a)&(tun_df["Pitch B"]==b))|((tun_df["Pitch A"]==b)&(tun_df["Pitch B"]==a))]
    if m.empty:
        return np.nan, "-"
    return m.iloc[0]["Tunnel Score"], m.iloc[0]["Grade"]


def _lookup_seq(setup, follow, seq_df):
    """Lookup sequence whiff% and chase% for a pitch pair. Returns (whiff%, chase%)."""
    if not isinstance(seq_df, pd.DataFrame) or seq_df.empty:
        return np.nan, np.nan
    m = seq_df[(seq_df["Setup Pitch"]==setup)&(seq_df["Follow Pitch"]==follow)]
    if m.empty:
        return np.nan, np.nan
    return m.iloc[0]["Whiff%"], m.iloc[0].get("Chase%", np.nan)


def _pitch_score_composite(pt_name, pt_data, hd, tun_df, platoon_label="Neutral", arsenal_data=None):
    """Unified composite score (0-100) for one pitch vs one hitter.
    Combines all available pitcher Trackman + hitter TrueMedia factors.

    pt_data: dict with our_whiff, our_csw, our_chase, our_ev_against, stuff_plus, etc.
    hd: dict with hitter data (whiff_2k_hard, whiff_2k_os, chase_pct, k_pct, etc.)
    tun_df: DataFrame of tunnel grades
    arsenal_data: dict with per-pitch arsenal info (for EffVelo, IVB, zone_eff)
    """
    components, weights = [], []
    is_hard = pt_name in _hard_pitches

    # 1. Stuff+ (13%) — raw pitch quality: 70-130 → 0-100
    sp = pt_data.get("stuff_plus", np.nan)
    if not pd.isna(sp):
        components.append(min(max((sp - 70) / 60 * 100, 0), 100)); weights.append(13)

    # 2. Our Whiff% (10%) — 0-50% → 0-100
    wh = pt_data.get("our_whiff", pt_data.get("whiff_pct", np.nan))
    if not pd.isna(wh):
        components.append(min(wh / 50 * 100, 100)); weights.append(10)

    # 3. Our CSW% (7%) — 0-40% → 0-100
    csw = pt_data.get("our_csw", pt_data.get("csw_pct", np.nan))
    if not pd.isna(csw):
        components.append(min(csw / 40 * 100, 100)); weights.append(7)

    # 4. Their 2K Whiff Rate (13%) — matched to pitch class (hard/offspeed) + our hand
    their_2k = hd.get("whiff_2k_hard" if is_hard else "whiff_2k_os", np.nan)
    if not pd.isna(their_2k):
        components.append(min(their_2k / 40 * 100, 100)); weights.append(13)

    # 5. Chase Exploitation (8%) — our chase generation × their chase tendency
    our_chase = pt_data.get("our_chase", pt_data.get("chase_pct", np.nan))
    their_chase = hd.get("chase_pct", np.nan)
    if not pd.isna(our_chase) and not pd.isna(their_chase):
        components.append(min((our_chase * their_chase) / (30 * 30) * 100, 100)); weights.append(8)

    # 6. Their Swing Rate vs Pitch Class (5%)
    sw_key = _swing_map.get(pt_name, "")
    their_sw = hd.get(sw_key, np.nan) if sw_key else np.nan
    if not pd.isna(their_sw):
        components.append(min(their_sw / 70 * 100, 100)); weights.append(5)

    # 7. Tunnel Score (7%) — BEST tunnel score involving this pitch (not average)
    if isinstance(tun_df, pd.DataFrame) and not tun_df.empty:
        t_m = tun_df[(tun_df["Pitch A"] == pt_name) | (tun_df["Pitch B"] == pt_name)]
        if not t_m.empty:
            components.append(t_m["Tunnel Score"].max()); weights.append(10)

    # 8. EV Against (5%) — penalize pitches we get hit hard on
    ev_ag = pt_data.get("our_ev_against", pt_data.get("ev_against", np.nan))
    if not pd.isna(ev_ag):
        components.append(min(max((96 - ev_ag) / 18 * 100, 0), 100)); weights.append(5)

    # 9. K-Prone Factor (4%) — high K hitters are exploitable
    k_pct = hd.get("k_pct", np.nan)
    if not pd.isna(k_pct):
        k_score = min(max((k_pct - 10) / 25 * 100, 0), 100)
        if not is_hard:
            k_score = min(k_score * 1.25, 100)
        components.append(k_score); weights.append(4)

    # 10. Platoon Factor (4%)
    plat_score = 50
    if "Adv" in platoon_label:
        plat_score = 80
    elif "Disadv" in platoon_label:
        plat_score = 25
    components.append(plat_score); weights.append(4)

    # 11. wOBA Split (6%) — how well hitter performs vs our hand
    woba_split = hd.get("woba_split", np.nan)
    if not pd.isna(woba_split):
        components.append(min(max((0.450 - woba_split) / 0.250 * 100, 0), 100)); weights.append(6)

    # 12. Hitter Contact% (3%) — LOW contact = more exploitable
    contact = hd.get("contact_pct", np.nan)
    if not pd.isna(contact):
        components.append(min(max((95 - contact) / 35 * 100, 0), 100)); weights.append(3)

    # 13. Hitter EV + Barrel weakness (3%)
    h_ev = hd.get("ev", np.nan)
    h_brl = hd.get("barrel_pct", np.nan)
    if not pd.isna(h_ev):
        ev_weak = min(max((95 - h_ev) / 17 * 100, 0), 100)
        if not pd.isna(h_brl):
            brl_weak = min(max((15 - h_brl) / 13 * 100, 0), 100)
            components.append((ev_weak + brl_weak) / 2); weights.append(3)
        else:
            components.append(ev_weak); weights.append(3)

    # 14. IVB (4%) — fastball-only: high IVB = harder to square up
    ars_pt = arsenal_data or {}
    ivb_val = ars_pt.get("ivb", pt_data.get("ivb", np.nan))
    if is_hard and not pd.isna(ivb_val):
        # 10" IVB → 0, 16" → 50, 22"+ → 100
        components.append(min(max((ivb_val - 10) / 12 * 100, 0), 100)); weights.append(4)

    # 15. EffVelo (3%) — higher effective velocity = harder to catch up
    eff_velo = ars_pt.get("eff_velo", np.nan)
    if not pd.isna(eff_velo):
        # 82 → 0, 88 → 50, 94+ → 100
        components.append(min(max((eff_velo - 82) / 12 * 100, 0), 100)); weights.append(3)

    # 16. Our Usage (7%) — pitches we actually throw should rank higher; low-usage pitches
    #     have small samples. But cap the penalty so quality secondaries can still surface.
    usage_pct = ars_pt.get("usage_pct", pt_data.get("usage", np.nan))
    if not pd.isna(usage_pct):
        # 0% → 0, 10% → 30, 20% → 50, 35%+ → 80, 55%+ → 100
        # Floor of 30 for any pitch >= 10% usage so secondaries aren't crushed
        raw_usage = min(max(usage_pct / 55 * 100, 0), 100)
        if usage_pct >= 10:
            raw_usage = max(raw_usage, 30)
        components.append(raw_usage); weights.append(7)

    # 18. Raw Velo (3%) — 93 mph FB should score higher than 86 mph; harder to react to
    raw_velo = ars_pt.get("avg_velo", pt_data.get("velo", np.nan))
    if not pd.isna(raw_velo):
        if is_hard:
            # Hard: 82 → 0, 88 → 50, 94+ → 100
            components.append(min(max((raw_velo - 82) / 12 * 100, 0), 100)); weights.append(3)
        else:
            # Offspeed: big velo diff from hard stuff is good; 70 → 30, 78 → 55, 85+ → 80
            components.append(min(max((85 - raw_velo) / 15 * 100, 10), 90)); weights.append(2)

    # 19. Barrel% Against (3%) — low barrel rate = effective pitch, penalize hittable pitches
    brl_ag = ars_pt.get("barrel_pct_against", np.nan)
    if not pd.isna(brl_ag):
        # 0% → 100, 5% → 67, 10% → 33, 15%+ → 0
        components.append(min(max((15 - brl_ag) / 15 * 100, 0), 100)); weights.append(3)

    # 20. Horizontal Break (3%) — offspeed only: more HB = more sweep/run = harder to barrel
    hb_val = ars_pt.get("hb", pt_data.get("hb", np.nan))
    if not is_hard and not pd.isna(hb_val):
        abs_hb = abs(hb_val)
        # 2" → 0, 8" → 50, 14"+ → 100
        components.append(min(max((abs_hb - 2) / 12 * 100, 0), 100)); weights.append(3)

    # 21. InZoneSwing% (2%) — aggressive in-zone swingers are more exploitable
    iz_swing = hd.get("iz_swing_pct", np.nan)
    if not pd.isna(iz_swing):
        # 55% → 0, 65% → 50, 75%+ → 100  (high = they swing a lot in zone = exploitable)
        components.append(min(max((iz_swing - 55) / 20 * 100, 0), 100)); weights.append(2)

    # 22. Extension (2%) — longer extension = closer release = more deception
    ext = ars_pt.get("extension", np.nan)
    if not pd.isna(ext):
        # 5.0 ft → 0, 6.0 → 50, 7.0+ → 100
        components.append(min(max((ext - 5.0) / 2.0 * 100, 0), 100)); weights.append(2)

    # 17. Zone Exploitation (5%) — cross our best zone with their zone weakness
    #     Formula: csw*0.6 + whiff*0.4, pitch-design multipliers (PZM),
    #     hitter exposure boosts from TrueMedia pitch location data.
    ze = ars_pt.get("zone_eff", {})
    if ze and hd:
        hitter_high = hd.get("high_pct", np.nan)
        hitter_low = hd.get("low_pct", np.nan)
        hitter_in = hd.get("inside_pct", np.nan)
        hitter_out = hd.get("outside_pct", np.nan)
        # Pitch-design zone multipliers — mirrors _get_pzm exactly (incl glove/arm)
        _ze_ivb = ars_pt.get("ivb", np.nan)
        if is_hard:
            if pt_name == "Sinker":
                _ze_pzm = {"up": 0.5, "down": 1.4, "chase_low": 1.3, "glove": 1.0, "arm": 1.0}
            elif not pd.isna(_ze_ivb) and _ze_ivb >= 16:
                _ze_pzm = {"up": 1.4, "down": 0.7, "chase_low": 0.4, "glove": 1.0, "arm": 1.0}
            elif not pd.isna(_ze_ivb) and _ze_ivb < 12:
                _ze_pzm = {"up": 0.7, "down": 1.3, "chase_low": 1.1, "glove": 1.0, "arm": 1.0}
            else:
                _ze_pzm = {"up": 1.15, "down": 0.9, "chase_low": 0.6, "glove": 1.0, "arm": 1.0}
        else:
            if pt_name in ("Curveball", "Knuckle Curve"):
                _ze_pzm = {"up": 0.2, "down": 1.2, "chase_low": 1.5, "glove": 0.8, "arm": 0.8}
            elif pt_name == "Changeup":
                _ze_pzm = {"up": 0.2, "down": 1.4, "chase_low": 1.4, "glove": 1.1, "arm": 1.0}
            else:
                _ze_pzm = {"up": 0.3, "down": 1.2, "chase_low": 1.3, "glove": 1.3, "arm": 0.8}
        # Map hitter inside/outside to pitcher arm/glove based on platoon
        _same = "Adv" in platoon_label
        _in_z = "arm" if _same else "glove"
        _out_z = "glove" if _same else "arm"
        best_exploit = 0
        for zn, zd in ze.items():
            if zd.get("n", 0) < 5:
                continue
            zone_whiff = zd.get("whiff_pct", 0) or 0
            zone_csw = zd.get("csw_pct", 0) or 0
            zone_quality = (zone_csw * 0.6 + zone_whiff * 0.4) * _ze_pzm.get(zn, 1.0)
            exposure_boost = 1.0
            if zn == "up" and not pd.isna(hitter_high) and hitter_high > 30:
                exposure_boost = 1.2
            elif zn == "down" and not pd.isna(hitter_low) and hitter_low > 35:
                exposure_boost = 1.15
            elif zn == _in_z and not pd.isna(hitter_in) and hitter_in > 28:
                exposure_boost = 1.15
            elif zn == _out_z and not pd.isna(hitter_out) and hitter_out > 28:
                exposure_boost = 1.15
            elif zn == "chase_low":
                chase_pct = hd.get("chase_pct", np.nan)
                if not pd.isna(chase_pct) and chase_pct > 28:
                    exposure_boost = 1.3
            best_exploit = max(best_exploit, zone_quality * exposure_boost)
        if best_exploit > 0:
            components.append(min(best_exploit / 40 * 100, 100)); weights.append(5)

    if not weights:
        return 50
    return sum(c * w for c, w in zip(components, weights)) / sum(weights)


def _build_3pitch_sequences(sorted_ps, hd, tun_df, seq_df):
    """Build best 3-pitch sequences: setup → bridge → putaway.
    HITTER-AWARE: Picks the putaway pitch based on the hitter's specific
    vulnerability, then finds the best setup/bridge path to get there.
    P1 (setup) must be a primary pitch (>= 15% usage) — you don't lead with
    a 10% sinker. Returns up to 3 sequences with different putaway pitches."""
    pitches = [name for name, data in sorted_ps if data.get("count", 0) >= 10]
    pitch_data = {name: data for name, data in sorted_ps if data.get("count", 0) >= 10}
    pitch_usage = {name: data.get("usage", 0) or 0 for name, data in sorted_ps}
    comp_scores = {name: data.get("score", 50) for name, data in sorted_ps}
    if len(pitches) < 2:
        return []
    # P1 candidates: must have meaningful usage (>= 15%), or fallback to top 2 by usage
    setup_candidates = [p for p in pitches if pitch_usage.get(p, 0) >= 15]
    if len(setup_candidates) < 2:
        setup_candidates = sorted(pitches, key=lambda p: pitch_usage.get(p, 0), reverse=True)[:2]

    # ── Step 1: Rank putaway candidates by hitter-specific vulnerability ──
    putaway_scores = {}
    for p in pitches:
        is_hard = p in _hard_pitches
        their_2k = hd.get("whiff_2k_hard" if is_hard else "whiff_2k_os", np.nan)
        comp = comp_scores.get(p, 50)
        whiff = pitch_data.get(p, {}).get("our_whiff", np.nan)
        chase = pitch_data.get(p, {}).get("our_chase", np.nan)
        score = comp * 0.50
        if not pd.isna(their_2k):
            score += min(their_2k / 40 * 100, 100) * 0.25
        if not pd.isna(whiff):
            score += min(whiff / 50 * 100, 100) * 0.15
        if not pd.isna(chase):
            score += min(chase / 40 * 100, 100) * 0.10
        putaway_scores[p] = score
    ranked_putaways = sorted(putaway_scores.items(), key=lambda x: x[1], reverse=True)

    # ── Step 2: For each putaway, find best setup → bridge path ──
    results = []
    for p3, p3_score in ranked_putaways:
        best_path = None
        best_path_score = -1
        for p2 in pitches:
            if p2 == p3:
                t_self, _ = _lookup_tunnel(p2, p3, tun_df)
                if pd.isna(t_self) or t_self <= 50:
                    continue
            for p1 in setup_candidates:  # P1 restricted to primary pitches
                if p1 == p2:
                    continue
                t12, g12 = _lookup_tunnel(p1, p2, tun_df)
                t23, g23 = _lookup_tunnel(p2, p3, tun_df)
                t12_bad = pd.isna(t12) or t12 < 25
                t23_bad = pd.isna(t23) or t23 < 25
                if t12_bad and t23_bad:
                    continue
                sw12, ch12 = _lookup_seq(p1, p2, seq_df)
                sw23, ch23 = _lookup_seq(p2, p3, seq_df)
                parts, wts = [], []
                # Tunnel quality (45%): t12 weight=18, t23 weight=27
                if not pd.isna(t12):
                    parts.append(t12); wts.append(18)
                if not pd.isna(t23):
                    parts.append(t23); wts.append(27)
                # Outcome effectiveness (40%): sw23=25, sw12=10, ch23=5
                if not pd.isna(sw23):
                    parts.append(min(sw23 / 50 * 100, 100)); wts.append(25)
                else:
                    parts.append(30); wts.append(25)
                if not pd.isna(sw12):
                    parts.append(min(sw12 / 50 * 100, 100)); wts.append(10)
                if not pd.isna(ch23):
                    parts.append(min(ch23 / 40 * 100, 100)); wts.append(5)
                # Pitch quality (15%): putaway composite=10, EffV gap=5
                parts.append(comp_scores.get(p3, 50)); wts.append(10)
                p1_effv = pitch_data.get(p1, {}).get("eff_velo", np.nan)
                p3_effv = pitch_data.get(p3, {}).get("eff_velo", np.nan)
                if not pd.isna(p1_effv) and not pd.isna(p3_effv):
                    gap = abs(p1_effv - p3_effv)
                    parts.append(min(25 + gap * 5, 100)); wts.append(5)
                else:
                    p1_velo = pitch_data.get(p1, {}).get("velo", np.nan)
                    p3_velo = pitch_data.get(p3, {}).get("velo", np.nan)
                    if not pd.isna(p1_velo) and not pd.isna(p3_velo):
                        gap = abs(p1_velo - p3_velo)
                        parts.append(min(25 + gap * 5, 100)); wts.append(5)
                if not wts:
                    continue
                path_score = sum(p*w for p,w in zip(parts, wts)) / sum(wts)
                if path_score > best_path_score:
                    best_path_score = path_score
                    ev_gap = abs(p1_effv - p3_effv) if not pd.isna(p1_effv) and not pd.isna(p3_effv) else np.nan
                    is_hard_p3 = p3 in _hard_pitches
                    their_2k = hd.get("whiff_2k_hard" if is_hard_p3 else "whiff_2k_os", np.nan)
                    best_path = {
                        "seq": f"{p1} → {p2} → {p3}", "p1": p1, "p2": p2, "p3": p3,
                        "score": round(p3_score * 0.35 + best_path_score * 0.65, 1),
                        "t12": t12, "t23": t23, "sw23": sw23, "their_2k": their_2k,
                        "effv_gap": ev_gap,
                    }
        if best_path:
            results.append(best_path)
        if len(results) >= 3:
            break
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def _build_4pitch_sequence(top_seqs, sorted_ps, hd, tun_df, seq_df):
    """Extend the best 3-pitch sequence into a 4-pitch sequence.
    Tries appending a P4 after P3 (deeper at-bat / secondary putaway) and
    also tries prepending a P0 before P1 (early-count setup). Returns the
    best 4-pitch dict or None if fewer than 3 pitches available."""
    if not top_seqs:
        return None
    pitches = [name for name, data in sorted_ps if data.get("count", 0) >= 10]
    pitch_data = {name: data for name, data in sorted_ps if data.get("count", 0) >= 10}
    comp_scores = {name: data.get("score", 50) for name, data in sorted_ps}
    if len(pitches) < 3:
        return None

    best_4 = None
    best_4_score = -1

    for seq3 in top_seqs[:2]:  # try extending top 2 three-pitch sequences
        p1, p2, p3 = seq3["p1"], seq3["p2"], seq3["p3"]
        base_score = seq3["score"]

        # ── Option A: append P4 after P3 ──
        for p4 in pitches:
            if p4 == p3 and p4 == p2:
                continue  # avoid 3 in a row of same pitch
            t34, g34 = _lookup_tunnel(p3, p4, tun_df)
            sw34, ch34 = _lookup_seq(p3, p4, seq_df)
            parts, wts = [], []
            # Tunnel t34 (35%)
            if not pd.isna(t34):
                parts.append(t34); wts.append(35)
            # Sequence whiff sw34 (30%)
            if not pd.isna(sw34):
                parts.append(min(sw34 / 50 * 100, 100)); wts.append(30)
            else:
                parts.append(30); wts.append(30)
            # P4 putaway composite + hitter vulnerability (20%): comp=10, t2k=10
            parts.append(comp_scores.get(p4, 50)); wts.append(10)
            is_hard_p4 = p4 in _hard_pitches
            t2k_p4 = hd.get("whiff_2k_hard" if is_hard_p4 else "whiff_2k_os", np.nan)
            if not pd.isna(t2k_p4):
                parts.append(min(t2k_p4 / 40 * 100, 100)); wts.append(10)
            # Chase ch34 (15%)
            if not pd.isna(ch34):
                parts.append(min(ch34 / 40 * 100, 100)); wts.append(15)
            if not wts:
                continue
            ext_score = sum(p * w for p, w in zip(parts, wts)) / sum(wts)
            total = base_score * 0.50 + ext_score * 0.50
            if total > best_4_score:
                best_4_score = total
                best_4 = {
                    "seq": f"{p1} → {p2} → {p3} → {p4}",
                    "p1": p1, "p2": p2, "p3": p3, "p4": p4,
                    "score": round(total, 1),
                    "sw34": sw34, "t34": t34,
                    "mode": "append",
                }

        # ── Option B: prepend P0 before P1 ──
        for p0 in pitches:
            if p0 == p1 and p0 == p2:
                continue
            t01, g01 = _lookup_tunnel(p0, p1, tun_df)
            sw01, ch01 = _lookup_seq(p0, p1, seq_df)
            parts, wts = [], []
            # Tunnel t01 (35%)
            if not pd.isna(t01):
                parts.append(t01); wts.append(35)
            # Sequence whiff sw01 (25%)
            if not pd.isna(sw01):
                parts.append(min(sw01 / 50 * 100, 100)); wts.append(25)
            # P0 quality composite (25%)
            parts.append(comp_scores.get(p0, 50)); wts.append(25)
            if not wts:
                continue
            ext_score = sum(p * w for p, w in zip(parts, wts)) / sum(wts)
            total = base_score * 0.55 + ext_score * 0.45
            if total > best_4_score:
                best_4_score = total
                best_4 = {
                    "seq": f"{p0} → {p1} → {p2} → {p3}",
                    "p1": p0, "p2": p1, "p3": p2, "p4": p3,
                    "score": round(total, 1),
                    "sw34": _lookup_seq(p2, p3, seq_df)[0], "t34": _lookup_tunnel(p2, p3, tun_df)[0],
                    "mode": "prepend",
                }

    return best_4


def _score_pitcher_vs_hitter(arsenal, hitter_profile):
    """Score how well our pitcher's arsenal exploits an opponent hitter's weaknesses.
    Uses the unified _pitch_score_composite for all per-pitch and overall scoring."""
    if arsenal is None or hitter_profile is None:
        return None
    throws = "R" if arsenal["throws"] == "Right" else "L"
    bats = hitter_profile["bats"]
    platoon_label = "Neutral"
    if bats == "S":
        platoon_label = "Switch (Neutral)"
    elif (throws == "L" and bats == "L") or (throws == "R" and bats == "R"):
        platoon_label = "Platoon Adv"
    elif (throws == "L" and bats == "R") or (throws == "R" and bats == "L"):
        platoon_label = "Platoon Disadv"
    woba_key = "woba_lhp" if throws == "L" else "woba_rhp"
    woba_split = hitter_profile.get(woba_key, np.nan)
    hand_key = "lhp" if throws == "L" else "rhp"

    # Build hitter_data dict (used by composite scorer and passed to UI)
    hd = {
        "pa": hitter_profile.get("pa", np.nan), "ops": hitter_profile.get("ops", np.nan),
        "woba": hitter_profile.get("woba", np.nan),
        "k_pct": hitter_profile.get("k_pct", np.nan), "bb_pct": hitter_profile.get("bb_pct", np.nan),
        "chase_pct": hitter_profile.get("chase_pct", np.nan), "swstrk_pct": hitter_profile.get("swstrk_pct", np.nan),
        "contact_pct": hitter_profile.get("contact_pct", np.nan),
        "swing_pct": hitter_profile.get("swing_pct", np.nan),
        "iz_swing_pct": hitter_profile.get("iz_swing_pct", np.nan),
        "p_per_pa": hitter_profile.get("p_per_pa", np.nan),
        "ev": hitter_profile.get("ev", np.nan), "barrel_pct": hitter_profile.get("barrel_pct", np.nan),
        "gb_pct": hitter_profile.get("gb_pct", np.nan), "fb_pct": hitter_profile.get("fb_pct", np.nan),
        "ld_pct": hitter_profile.get("ld_pct", np.nan),
        "pull_pct": hitter_profile.get("pull_pct", np.nan),
        "woba_split": woba_split,
        "whiff_2k_hard": hitter_profile.get(f"whiff_2k_{hand_key}_hard", np.nan),
        "whiff_2k_os": hitter_profile.get(f"whiff_2k_{hand_key}_os", np.nan),
        "fp_swing_hard": hitter_profile.get("fp_swing_hard_empty", np.nan),
        "fp_swing_ch": hitter_profile.get("fp_swing_ch_empty", np.nan),
        "swing_vs_hard": hitter_profile.get("swing_vs_hard", np.nan),
        "swing_vs_sl": hitter_profile.get("swing_vs_sl", np.nan),
        "swing_vs_cb": hitter_profile.get("swing_vs_cb", np.nan),
        "swing_vs_ch": hitter_profile.get("swing_vs_ch", np.nan),
        "high_pct": hitter_profile.get("high_pct", np.nan),
        "low_pct": hitter_profile.get("low_pct", np.nan),
        "inside_pct": hitter_profile.get("inside_pct", np.nan),
        "outside_pct": hitter_profile.get("outside_pct", np.nan),
    }

    # Get tunnel data from arsenal
    tun_df = arsenal.get("tunnels", pd.DataFrame())

    # Score each pitch using the unified composite scorer
    pitch_scores = {}
    for pt_name, pt_data in arsenal["pitches"].items():
        is_hard = pt_name in _hard_pitches
        # Build pt_data dict compatible with composite scorer
        pd_compat = {
            "stuff_plus": pt_data.get("stuff_plus", np.nan),
            "our_whiff": pt_data.get("whiff_pct", np.nan),
            "our_csw": pt_data.get("csw_pct", np.nan),
            "our_chase": pt_data.get("chase_pct", np.nan),
            "our_ev_against": pt_data.get("ev_against", np.nan),
        }
        # Compute composite score
        comp_score = _pitch_score_composite(
            pt_name, pd_compat, hd, tun_df, platoon_label,
            arsenal_data=pt_data  # pass full arsenal pitch data for IVB/EffVelo/zone_eff
        )
        # Build reasons from notable factors
        reasons = []
        stuff = pt_data.get("stuff_plus", np.nan)
        if not pd.isna(stuff) and stuff >= 115:
            reasons.append(f"elite stuff ({stuff:.0f} S+)")
        our_whiff = pt_data.get("whiff_pct", np.nan)
        if not pd.isna(our_whiff) and our_whiff >= 35:
            reasons.append(f"high whiff ({our_whiff:.0f}%)")
        whiff_2k = hitter_profile.get(f"whiff_2k_{hand_key}_{'hard' if is_hard else 'os'}", np.nan)
        if not pd.isna(whiff_2k) and whiff_2k > 35:
            reasons.append(f"hitter whiffs {whiff_2k:.0f}% on 2K {'hard' if is_hard else 'offspeed'}")
        hitter_chase = hitter_profile.get("chase_pct", np.nan)
        our_chase = pt_data.get("chase_pct", np.nan)
        if not pd.isna(hitter_chase) and not pd.isna(our_chase) and hitter_chase > 28 and our_chase > 30:
            reasons.append(f"chaser ({hitter_chase:.0f}%) + our chase gen ({our_chase:.0f}%)")
        our_ev_against = pt_data.get("ev_against", np.nan)
        if not pd.isna(our_ev_against) and our_ev_against > 90:
            reasons.append(f"gets hit hard ({our_ev_against:.1f} EV against)")

        pitch_scores[pt_name] = {
            "score": round(comp_score, 1), "reasons": reasons,
            "our_whiff": our_whiff, "our_chase": our_chase,
            "our_csw": pt_data.get("csw_pct", np.nan),
            "our_ev_against": pt_data.get("ev_against", np.nan),
            "stuff_plus": stuff, "usage": pt_data["usage_pct"],
            "velo": pt_data["avg_velo"],
            "eff_velo": pt_data.get("eff_velo", np.nan),
            "spin": pt_data.get("avg_spin", np.nan),
            "ivb": pt_data.get("ivb", np.nan),
            "hb": pt_data.get("hb", np.nan),
            "count": pt_data.get("count", 0),
        }

    sorted_pitches = sorted(pitch_scores.items(), key=lambda x: x[1]["score"], reverse=True)
    recommendations = []
    if sorted_pitches:
        best = sorted_pitches[0]
        recommendations.append(f"Lead with **{best[0]}** ({best[1]['score']:.0f})")
        offspeed = [(n, d) for n, d in sorted_pitches if n not in _hard_pitches]
        if offspeed:
            recommendations.append(f"Putaway: **{offspeed[0][0]}** ({offspeed[0][1]['score']:.0f})")
        elif len(sorted_pitches) > 1:
            recommendations.append(f"Secondary: **{sorted_pitches[1][0]}** ({sorted_pitches[1][1]['score']:.0f})")
    # Discipline-based notes
    hitter_bb = hitter_profile.get("bb_pct", np.nan)
    hitter_chase_pct = hitter_profile.get("chase_pct", np.nan)
    if not pd.isna(hitter_bb) and hitter_bb < 5:
        recommendations.append("Free swinger — attack the zone early")
    elif not pd.isna(hitter_bb) and hitter_bb > 14:
        recommendations.append("Patient hitter — don't nibble, pitch to contact")
    if not pd.isna(hitter_chase_pct) and hitter_chase_pct > 35:
        recommendations.append(f"Chases aggressively ({hitter_chase_pct:.0f}%) — expand early")
    gb_pct = hitter_profile.get("gb_pct", np.nan)
    if not pd.isna(gb_pct) and gb_pct > 55:
        recommendations.append(f"Ground ball hitter ({gb_pct:.0f}% GB) — pitch up in zone")
    if not pd.isna(woba_split) and woba_split >= 0.380:
        recommendations.append(f"⚠ Danger — {woba_split:.3f} wOBA vs {'LHP' if throws=='L' else 'RHP'}")
    elif not pd.isna(woba_split) and woba_split <= 0.260:
        recommendations.append(f"Exploitable — {woba_split:.3f} wOBA vs {'LHP' if throws=='L' else 'RHP'}")

    # Usage-weighted overall score from composite
    if pitch_scores:
        total_usage = sum(v["usage"] for v in pitch_scores.values())
        if total_usage > 0:
            overall = sum(v["score"] * v["usage"] for v in pitch_scores.values()) / total_usage
        else:
            overall = np.mean([v["score"] for v in pitch_scores.values()])
    else:
        overall = 50
    return {
        "hitter": hitter_profile["name"], "pitcher": arsenal["name"],
        "bats": bats, "platoon": platoon_label,
        "overall_score": overall,
        "pitch_scores": pitch_scores, "recommendations": recommendations,
        "hitter_data": hd,
    }


def _score_hitter_vs_pitcher(hitter_tm, pitcher_profile):
    """Score hitter vs pitcher using transparent per-pitch Edge system.
    Each pitch gets Advantage / Neutral / Vulnerable with readable reasons."""
    if hitter_tm is None or pitcher_profile is None:
        return None
    bats = hitter_tm["bats"]
    throws = pitcher_profile["throws"]
    platoon_factor, platoon_label = 1.0, "Neutral"
    if (throws == "R" and bats == "Left") or (throws == "L" and bats == "Right"):
        platoon_factor, platoon_label = 1.08, "Platoon Adv"
    elif (throws == "R" and bats == "Right") or (throws == "L" and bats == "Left"):
        platoon_factor, platoon_label = 0.92, "Same-Side"
    elif bats == "Switch":
        platoon_factor, platoon_label = 1.04, "Switch"

    _hard_set = {"Fastball", "Sinker", "Cutter"}
    zones = hitter_tm.get("zones", {})
    zone_grid = hitter_tm.get("zone_grid", {})

    def _timing_note(depth):
        if pd.isna(depth):
            return ""
        if depth > 0.5:
            return f"Out front +{depth:.1f}"
        if depth < -1.5:
            return f"Deep {depth:.1f}"
        return ""

    # 2A. Zone-weighted EV: cross per-pitch zone_damage with pitcher location %s
    def _zone_weighted_ev(our_pitch_data, pp):
        """Compute zone-weighted EV for a pitch type using pitcher location tendencies.
        Returns weighted EV or NaN if insufficient data.
        Only uses zone weighting when we have BOTH vertical and horizontal location data."""
        zd = our_pitch_data.get("zone_damage", {}) if our_pitch_data else {}
        if len(zd) < 2:
            return np.nan
        h_pct = pp.get("loc_high_pct", np.nan)
        vmid_pct = pp.get("loc_vmid_pct", np.nan)
        l_pct = pp.get("loc_low_pct", np.nan)
        in_pct = pp.get("loc_inside_pct", np.nan)
        out_pct = pp.get("loc_outside_pct", np.nan)
        # Only weight by location if we have real data — don't fabricate weights
        has_vert = not pd.isna(h_pct) or not pd.isna(l_pct)
        has_horiz = not pd.isna(in_pct) or not pd.isna(out_pct)
        if not has_vert and not has_horiz:
            # No location data — just average the zone EVs equally
            evs = [zdata.get("avg_ev", np.nan) for zdata in zd.values()]
            valid = [e for e in evs if not pd.isna(e)]
            return np.mean(valid) if valid else np.nan
        # Use actual location data, fill missing with uniform (33/50)
        v_h = h_pct if not pd.isna(h_pct) else 33
        v_m = vmid_pct if not pd.isna(vmid_pct) else 34
        v_l = l_pct if not pd.isna(l_pct) else 33
        h_i = in_pct if not pd.isna(in_pct) else 50
        h_o = out_pct if not pd.isna(out_pct) else 50
        zone_weights = {
            "up_in": v_h * h_i / 100, "up_out": v_h * h_o / 100,
            "mid_in": v_m * h_i / 100, "mid_out": v_m * h_o / 100,
            "down_in": v_l * h_i / 100, "down_out": v_l * h_o / 100,
        }
        weighted_ev, total_w = 0.0, 0.0
        for zkey, zdata in zd.items():
            ev = zdata.get("avg_ev", np.nan)
            if pd.isna(ev):
                continue
            w = zone_weights.get(zkey, 1.0)
            weighted_ev += ev * w
            total_w += w
        return weighted_ev / total_w if total_w > 0 else np.nan

    # 2B. Upgraded zone_cross: use per-pitch zone_damage first, fall back to zone_grid
    def _zone_cross_note(opp_pitch):
        """Cross pitcher location tendency with our per-pitch zone_damage or zone_grid."""
        h_pct = pitcher_profile.get("loc_high_pct", np.nan)
        l_pct = pitcher_profile.get("loc_low_pct", np.nan)
        # Find dominant vertical tendency
        best_tend, best_pct = "", 0
        if not pd.isna(h_pct) and h_pct > best_pct:
            best_tend, best_pct = "high", h_pct
        if not pd.isna(l_pct) and l_pct > best_pct:
            best_tend, best_pct = "low", l_pct
        if not best_tend or best_pct < 25:
            return ""
        # Map "high"/"low" to zone_damage keys
        zd_vert = "up" if best_tend == "high" else "down"
        # Try per-pitch zone_damage first
        our_pt_data = hitter_tm["by_pitch_type"].get(opp_pitch, {})
        zd = our_pt_data.get("zone_damage", {}) if our_pt_data else {}
        zd_cells = {k: v for k, v in zd.items() if k.startswith(zd_vert)}
        if zd_cells:
            best_zd_key = max(zd_cells, key=lambda k: zd_cells[k].get("avg_ev", 0)
                              if not pd.isna(zd_cells[k].get("avg_ev", np.nan)) else 0)
            best_zd = zd_cells[best_zd_key]
            cell_ev = best_zd.get("avg_ev", np.nan)
            barrels = best_zd.get("barrels", 0)
            if not pd.isna(cell_ev):
                loc_label = best_zd_key.replace("_", " ")
                pitch_label = opp_pitch.lower() + "s" if not opp_pitch.endswith("er") else opp_pitch.lower()
                if barrels >= 2:
                    return (f"They throw {pitch_label} {best_tend} ({best_pct:.0f}%) "
                            f"\u2014 we barrel {pitch_label} {loc_label} ({cell_ev:.0f} EV, {barrels} barrels)")
                ev_label = "weak" if cell_ev < 82 else ""
                return (f"They throw {pitch_label} {best_tend} ({best_pct:.0f}%) "
                        f"\u2014 we hit {loc_label} {pitch_label} at {cell_ev:.0f} EV"
                        + (f" ({ev_label})" if ev_label else ""))
        # Fall back to overall zone_grid
        match_keys = [k for k in zone_grid if best_tend.capitalize() in k.split("_")[0]]
        if not match_keys:
            return ""
        best_cell = max(match_keys, key=lambda k: zone_grid[k].get("avg_ev", 0)
                        if not pd.isna(zone_grid[k].get("avg_ev", np.nan)) else 0)
        cell_ev = zone_grid[best_cell].get("avg_ev", np.nan)
        if pd.isna(cell_ev):
            return ""
        label = best_cell.replace("_", " ").lower()
        return f"They go {best_tend} ({best_pct:.0f}%) \u2014 we hit {label} ({cell_ev:.0f} EV)"

    pitch_edges = []
    weighted_score, total_weight = 0.0, 0.0
    approach_notes = []

    for opp_pitch, opp_usage in pitcher_profile["pitch_mix"].items():
        if opp_usage < 5:
            continue
        our_data = hitter_tm["by_pitch_type"].get(opp_pitch, {})
        # 2A: Use zone-weighted EV for edge classification, keep raw for display
        our_ev_raw = our_data.get("avg_ev", np.nan) if our_data else np.nan
        our_ev_zw = _zone_weighted_ev(our_data, pitcher_profile)
        our_ev = our_ev_zw if not pd.isna(our_ev_zw) else our_ev_raw
        # Fallback to overall EV if no per-pitch data
        overall_ev = hitter_tm.get("overall", {}).get("avg_ev", np.nan)
        if pd.isna(our_ev) and not pd.isna(overall_ev):
            our_ev = overall_ev
        our_whiff = our_data.get("whiff_pct", np.nan) if our_data else np.nan
        our_barrel = our_data.get("barrel_pct", np.nan) if our_data else np.nan
        our_depth = our_data.get("contact_depth", np.nan) if our_data else np.nan
        their_whiff = pitcher_profile.get("swstrk_pct", np.nan)
        their_chase = pitcher_profile.get("chase_pct", np.nan)

        # Edge classification — barrel-aware, sample-size-sensitive
        has_pitch_data = bool(our_data)  # True if we have pitch-specific data
        ip_n = our_data.get("seen", 0) if our_data else 0
        # Use InPlay-based EV only if we have real per-pitch data (not overall fallback)
        ev_is_real = has_pitch_data and not pd.isna(our_ev_raw)
        ev_ok = ev_is_real and our_ev >= 88
        ev_great = ev_is_real and our_ev >= 90
        whiff_ok = not pd.isna(our_whiff) and our_whiff <= 25
        whiff_bad = not pd.isna(our_whiff) and our_whiff >= 35
        ev_bad = ev_is_real and our_ev < 80
        ev_weak = ev_is_real and our_ev < 85  # not terrible but not good either
        barrel_good = not pd.isna(our_barrel) and our_barrel >= 8
        their_whiff_high = not pd.isna(their_whiff) and their_whiff >= 25

        if (ev_ok and whiff_ok) or (ev_great and barrel_good):
            edge = "Advantage"
        elif (whiff_bad and ev_bad) or (ev_bad and their_whiff_high) or (not ev_is_real and whiff_bad) or (whiff_bad and ev_weak):
            edge = "Vulnerable"
        else:
            edge = "Neutral"

        # Build reason string
        if edge == "Advantage":
            parts = []
            if barrel_good:
                parts.append(f"We barrel it ({our_barrel:.0f}%)")
            if ev_is_real:
                parts.append(f"hit it hard ({our_ev:.0f} EV)")
            if not pd.isna(our_whiff):
                parts.append(f"{our_whiff:.0f}% whiff")
            reason = " and ".join(parts[:2]) if parts else "Strong matchup"
        elif edge == "Vulnerable":
            parts = []
            if whiff_bad:
                parts.append(f"High whiff ({our_whiff:.0f}%)")
            tn = _timing_note(our_depth)
            if tn:
                parts.append(f"we're {tn.lower()} on it")
            if ev_bad:
                parts.append(f"weak contact ({our_ev:.0f} EV)")
            reason = " \u2014 ".join(parts) if parts else "Vulnerable"
        else:
            ev_str = f"{our_ev:.0f} EV" if not pd.isna(our_ev) else "no data"
            wh_str = f"{our_whiff:.0f}% whiff" if not pd.isna(our_whiff) else ""
            reason_parts = [ev_str]
            if wh_str:
                reason_parts.append(wh_str)
            reason = f"Neutral \u2014 " + ", ".join(reason_parts)

        edge_val = {"Advantage": 70, "Neutral": 50, "Vulnerable": 30}[edge]
        weight = opp_usage / 100.0

        pitch_edge = {
            "pitch": opp_pitch,
            "usage": opp_usage,
            "edge": edge,
            "our_ev": our_ev,
            "our_ev_raw": our_ev_raw,
            "our_whiff": our_whiff,
            "our_barrel": our_barrel if not pd.isna(our_barrel) else np.nan,
            "their_whiff": their_whiff,
            "their_chase": their_chase,
            "timing_note": _timing_note(our_depth),
            "zone_cross": _zone_cross_note(opp_pitch),
            "reason": reason,
        }
        pitch_edges.append(pitch_edge)
        weighted_score += edge_val * weight
        total_weight += weight

    # Overall score with platoon adjustment
    raw_score = weighted_score / total_weight if total_weight > 0 else 50
    overall = max(0, min(100, raw_score * platoon_factor))

    # Build approach notes from edge data
    adv_pitches = [pe for pe in pitch_edges if pe["edge"] == "Advantage"]
    vuln_pitches = [pe for pe in pitch_edges if pe["edge"] == "Vulnerable"]
    if adv_pitches:
        # Weight by EV × usage to pick the most impactful Advantage pitch
        best = max(adv_pitches, key=lambda pe: (pe["our_ev"] if not pd.isna(pe["our_ev"]) else 80) * (pe["usage"] / 100))
        approach_notes.append(f"Sit on **{best['pitch']}** ({best['usage']:.0f}% of mix)")
    if vuln_pitches:
        worst = max(vuln_pitches, key=lambda pe: pe["our_whiff"] if not pd.isna(pe["our_whiff"]) else 0)
        approach_notes.append(f"Lay off **{worst['pitch']}** out of zone")
    unknown = [pe for pe in pitch_edges if pd.isna(pe["our_ev"]) and pe["usage"] > 10]
    if unknown:
        approach_notes.append(f"No Trackman data vs {', '.join(pe['pitch'] for pe in unknown)} \u2014 be ready")
    loc_tend = pitcher_profile.get("location_tendency", "")
    if loc_tend and loc_tend not in ("balanced", "unknown"):
        approach_notes.append(f"Lives {loc_tend}")

    # Build pitch_details dict for backward compatibility
    pitch_details = {}
    for pe in pitch_edges:
        pitch_details[pe["pitch"]] = {
            "score": {"Advantage": 70, "Neutral": 50, "Vulnerable": 30}[pe["edge"]],
            "usage": pe["usage"],
            "edge": pe["edge"],
            "our_ev": pe["our_ev"],
            "our_whiff": pe["our_whiff"],
            "our_barrel": pe["our_barrel"],
            "timing_note": pe["timing_note"],
            "zone_cross": pe["zone_cross"],
            "reason": pe["reason"],
        }

    return {
        "hitter": hitter_tm["name"], "pitcher": pitcher_profile["name"],
        "bats": bats, "platoon": platoon_label,
        "overall_score": round(overall, 1),
        "pitch_edges": pitch_edges,
        "pitch_details": pitch_details, "approach_notes": approach_notes,
    }


def _generate_at_bat_script(hitter_profile, pitcher_profile, matchup_result):
    """Generate at-bat script with tag line, 1P/ahead/2-strike plans, and pitch edges."""
    if not matchup_result or not pitcher_profile.get("pitch_mix"):
        return None
    _hard_set = {"Fastball", "Sinker", "Cutter"}
    mix = pitcher_profile["pitch_mix"]
    real_mix = {p: u for p, u in mix.items() if u >= 5}
    if not real_mix:
        return None
    zones = hitter_profile.get("zones", {})
    by_class = hitter_profile.get("by_pitch_class", {})
    pitch_edges = matchup_result.get("pitch_edges", [])
    pitch_details = matchup_result.get("pitch_details", {})
    sp = hitter_profile.get("swing_path") or {}
    disc = hitter_profile.get("discipline") or {}
    fp_data = hitter_profile.get("first_pitch") or {}
    two_k_data = hitter_profile.get("two_strike") or {}
    zone_grid = hitter_profile.get("zone_grid") or {}
    by_pt = hitter_profile.get("by_pitch_type", {})
    # 3A: New V3 fields
    barrel_zones = hitter_profile.get("barrel_zones", [])
    spray = hitter_profile.get("spray", {})
    two_k_whiff_hard = two_k_data.get("whiff_hard", np.nan) if two_k_data else np.nan
    two_k_whiff_os = two_k_data.get("whiff_os", np.nan) if two_k_data else np.nan
    count_ev = hitter_profile.get("count_ev_grid", {})

    sorted_mix = sorted(real_mix.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_mix[0] if sorted_mix else (None, 0)
    hard_pitches = [(p, u) for p, u in sorted_mix if p in _hard_set]
    os_pitches = [(p, u) for p, u in sorted_mix if p not in _hard_set]

    # ── Helpers ──
    def _best_damage_zone(zg, pp):
        """Find our best zone_grid cell, with pitcher tendency as small tiebreaker.
        EV is the primary sort — pitcher location only nudges between similar-EV zones."""
        h_pct = pp.get("loc_high_pct", np.nan)
        l_pct = pp.get("loc_low_pct", np.nan)
        candidates = []
        for key, cell in zg.items():
            if cell.get("n", 0) < 5:
                continue
            ev = cell.get("avg_ev", np.nan)
            if pd.isna(ev):
                continue
            parts = key.split("_")
            row = parts[0] if parts else ""
            # Pitcher tendency is a SMALL tiebreaker (max +3 EV boost), not dominant
            bonus = 0.0
            if "High" in row and not pd.isna(h_pct) and h_pct >= 25:
                bonus = min(h_pct / 30, 3.0)  # max 3 EV bonus
            if "Low" in row and not pd.isna(l_pct) and l_pct >= 25:
                bonus = min(l_pct / 30, 3.0)
            candidates.append((key, ev, ev + bonus))
        if not candidates:
            return None, np.nan
        best = max(candidates, key=lambda c: c[2])
        return best[0], best[1]  # return real EV, not boosted

    def _chase_vulnerability(z):
        """Check all 4 chase zones. Return list of (zone_label, whiff%) where whiff > 35%."""
        vulns = []
        for cz in ["chase_up", "chase_down", "chase_in", "chase_away"]:
            zd = z.get(cz, {})
            wh = zd.get("whiff_pct", np.nan)
            if zd.get("n", 0) >= 5 and not pd.isna(wh) and wh > 35:
                vulns.append((cz.replace("chase_", ""), wh))
        vulns.sort(key=lambda x: x[1], reverse=True)
        return vulns

    def _timing_note_str(depth):
        if pd.isna(depth):
            return ""
        if depth > 0.5:
            return f"early +{depth:.1f}"
        if depth < -1.5:
            return f"deep {depth:.1f}"
        return ""

    # ── Section 0: TAG LINE ──
    swing_type = sp.get("swing_type", "Unknown")
    bs_avg = sp.get("bat_speed_avg")
    chase = disc.get("chase_pct", np.nan)
    z_contact = disc.get("z_contact_pct", np.nan)
    if not pd.isna(chase) and not pd.isna(z_contact):
        disc_score = (100 - chase) * 0.5 + z_contact * 0.5
        disc_grade = "A" if disc_score >= 80 else "B" if disc_score >= 65 else "C" if disc_score >= 50 else "D"
    else:
        disc_grade = "-"
    overall_ev = hitter_profile.get("overall", {}).get("avg_ev", np.nan)
    overall_barrel = hitter_profile.get("overall", {}).get("barrel_pct", np.nan)
    overall_hh = hitter_profile.get("overall", {}).get("hard_hit_pct", np.nan)
    overall_ss = hitter_profile.get("overall", {}).get("sweet_spot_pct", np.nan)
    attack_angle = sp.get("attack_angle")
    tag_parts = [f"{hitter_profile['bats'][0]}HH"]
    if not pd.isna(overall_ev):
        tag_parts.append(f"{overall_ev:.0f} EV")
    if not pd.isna(overall_barrel) and overall_barrel >= 3:
        tag_parts.append(f"{overall_barrel:.0f}% barrel")
    if swing_type != "Unknown":
        tag_parts.append(swing_type)
    if bs_avg is not None:
        tag_parts.append(f"{bs_avg:.0f} mph bat")
    tag_parts.append(f"{disc_grade} discipline")
    if not pd.isna(chase):
        tag_parts.append(f"{chase:.0f}% chase")
    tag_line = " \u2022 ".join(tag_parts)

    # ── Section 1: SCOUTING THIS PITCHER ──
    scout_lines = []
    for p, u in sorted_mix[:3]:
        loc_hint = ""
        if p in _hard_set:
            h_pct = pitcher_profile.get("loc_high_pct", np.nan)
            if not pd.isna(h_pct) and h_pct >= 28:
                loc_hint = "up"
            else:
                l_pct = pitcher_profile.get("loc_low_pct", np.nan)
                loc_hint = "low" if (not pd.isna(l_pct) and l_pct >= 30) else ""
        else:
            l_pct = pitcher_profile.get("loc_low_pct", np.nan)
            loc_hint = "low" if (not pd.isna(l_pct) and l_pct >= 28) else "arm-side"
        velo = pitcher_profile.get("velo", np.nan)
        velo_str = f" {velo:.0f}mph" if not pd.isna(velo) and p == sorted_mix[0][0] else ""
        scout_lines.append(f"{p}{velo_str} ({u:.0f}%) {loc_hint}".strip())
    comploc = pitcher_profile.get("comploc_pct", np.nan)
    if not pd.isna(comploc) and comploc < 40:
        scout_lines.append(f"Low command ({comploc:.0f}%)")

    # ── Section 2: FIRST PITCH ──
    fp_pitch = primary[0]
    fp_usage = primary[1]
    callstrk = pitcher_profile.get("callstrk_pct", np.nan)
    our_fp_swing = fp_data.get("swing_pct", np.nan) if fp_data else np.nan
    our_fp_ev = fp_data.get("avg_ev", np.nan) if fp_data else np.nan
    our_1p_whiff = by_pt.get(fp_pitch, {}).get("whiff_pct", np.nan)
    our_1p_ev = by_pt.get(fp_pitch, {}).get("avg_ev", np.nan)

    # Pitcher's 1P location tendency
    h_pct = pitcher_profile.get("loc_high_pct", np.nan)
    l_pct = pitcher_profile.get("loc_low_pct", np.nan)
    fp_loc_hint = ""
    if not pd.isna(h_pct) and h_pct >= 28:
        fp_loc_hint = " up"
    elif not pd.isna(l_pct) and l_pct >= 30:
        fp_loc_hint = " low"

    fp_expect = f"{fp_pitch}{fp_loc_hint} ({fp_usage:.0f}%)"

    # Decision tree
    # Check if primary pitch is our Advantage pitch (barrel it / crush it) — override whiff gate
    fp_edge = pitch_details.get(fp_pitch, {}).get("edge", "")
    our_1p_barrel = by_pt.get(fp_pitch, {}).get("barrel_pct", np.nan)
    fp_is_advantage = fp_edge == "Advantage"
    if fp_is_advantage and not pd.isna(our_1p_ev) and our_1p_ev >= 87:
        fp_plan = "GREEN LIGHT"
        ev_str = f"{our_1p_ev:.0f} EV"
        barrel_str = f", {our_1p_barrel:.0f}% barrel" if not pd.isna(our_1p_barrel) and our_1p_barrel >= 5 else ""
        wh_str = f", {our_1p_whiff:.0f}% whiff" if not pd.isna(our_1p_whiff) else ""
        fp_detail = f"Advantage pitch \u2014 {ev_str}{barrel_str}{wh_str}"
    elif not pd.isna(our_1p_whiff) and our_1p_whiff > 30:
        fp_plan = "BE SELECTIVE"
        fp_detail = f"High whiff on {fp_pitch} ({our_1p_whiff:.0f}%) \u2014 take unless perfect"
    elif not pd.isna(our_fp_ev) and our_fp_ev > 87 and not pd.isna(our_fp_swing) and our_fp_swing >= 30:
        fp_plan = "GREEN LIGHT"
        ev_str = f"{our_1p_ev:.0f} EV" if not pd.isna(our_1p_ev) else f"{our_fp_ev:.0f} EV"
        wh_str = f", {our_1p_whiff:.0f}% whiff" if not pd.isna(our_1p_whiff) else ""
        fp_detail = f"{ev_str}{wh_str}"
    elif not pd.isna(callstrk) and callstrk > 18:
        fp_plan = "SWING IF IN ZONE"
        fp_detail = f"High called-strike rate ({callstrk:.0f}%)"
    else:
        fp_plan = "SEE IT FIRST"
        fp_detail = "Gather info, swing only if middle-middle"

    # Look zone: cross best damage zone with pitcher 1P tendency
    dmg_key, dmg_ev = _best_damage_zone(zone_grid, pitcher_profile)
    fp_look = ""
    if dmg_key and not pd.isna(dmg_ev) and dmg_ev > 85:
        fp_look = f"{dmg_key.replace('_', '-').lower()} ({dmg_ev:.0f} EV damage zone)"

    first_pitch = {
        "expect": fp_expect,
        "plan": fp_plan,
        "detail": fp_detail,
        "look": fp_look,
    }

    # ── Section 3: AHEAD IN COUNT ──
    adv_edges = [pe for pe in pitch_edges if pe["edge"] == "Advantage"]
    if adv_edges:
        best_adv = max(adv_edges, key=lambda pe: pe["our_ev"] if not pd.isna(pe["our_ev"]) else 0)
        sit_pitch = best_adv["pitch"]
        sit_ev = best_adv["our_ev"]
        sit_str = f"Sit on: {sit_pitch} \u2014 we crush it ({sit_ev:.0f} EV)" if not pd.isna(sit_ev) else f"Sit on: {sit_pitch}"
    else:
        sit_str = "Be aggressive in zone \u2014 no clear Advantage pitch"

    # Hunt zone: our best damage zone where pitcher also throws
    hunt_key, hunt_ev = _best_damage_zone(zone_grid, pitcher_profile)
    hunt_str = ""
    if hunt_key and not pd.isna(hunt_ev) and hunt_ev > 85:
        hunt_str = f"{hunt_key.replace('_', '-').lower()} ({hunt_ev:.0f} EV)"

    # 3B: Barrel hunt zone — always show top barrel zone, add pitcher overlap context
    barrel_hunt_str = ""
    if barrel_zones:
        h_pct = pitcher_profile.get("loc_high_pct", np.nan)
        l_pct = pitcher_profile.get("loc_low_pct", np.nan)
        in_pct = pitcher_profile.get("loc_inside_pct", np.nan)
        out_pct = pitcher_profile.get("loc_outside_pct", np.nan)
        # Always use top barrel zone (most barrels)
        bz_key, bz_barrels, bz_ev = barrel_zones[0]
        bz_lower = bz_key.lower()
        zone_label = bz_key.replace("_", " ")
        # Check if pitcher throws to this zone
        overlap = False
        if "high" in bz_lower and not pd.isna(h_pct) and h_pct >= 25:
            overlap = True
        if "low" in bz_lower and not pd.isna(l_pct) and l_pct >= 25:
            overlap = True
        if ("inside" in bz_lower or "far_in" in bz_lower) and not pd.isna(in_pct) and in_pct >= 25:
            overlap = True
        if ("outside" in bz_lower or "far_out" in bz_lower) and not pd.isna(out_pct) and out_pct >= 25:
            overlap = True
        if overlap:
            barrel_hunt_str = (f"Barrels come {zone_label.lower()} ({bz_barrels} barrels, {bz_ev:.0f} EV) "
                               f"\u2014 pitcher throws there. HUNT {zone_label.upper()}.")
        else:
            barrel_hunt_str = (f"Barrels come {zone_label.lower()} ({bz_barrels} barrels, {bz_ev:.0f} EV). "
                               f"HUNT {zone_label.upper()} \u2014 force your pitch.")

    # 3C: Count-aware EV callouts — AHEAD
    count_ahead_note = ""
    for ck in ["2-0", "3-1", "3-0"]:
        cd = count_ev.get(ck, {})
        cev = cd.get("avg_ev", np.nan)
        cn = cd.get("n", 0)
        if not pd.isna(cev) and cev >= 90 and cn >= 5:
            count_ahead_note = f"You mash {ck} ({cev:.0f} EV) \u2014 sit dead red"
            break

    when_ahead = {
        "sit_on": sit_str,
        "hunt_zone": hunt_str,
        "barrel_hunt": barrel_hunt_str,
        "count_note": count_ahead_note,
    }

    # ── Section 3B: WHEN BEHIND ──
    behind_note = ""
    # When behind (0-1, 0-2, 1-2), pitcher controls — what should hitter do?
    behind_pitches = []
    for p, u in sorted_mix:
        if p not in _hard_set and u >= 10:
            behind_pitches.append((p, u))
    # Add location context from pitcher profile
    low_pct_b = pitcher_profile.get("loc_low_pct", np.nan)
    loc_hint_b = ""
    if not pd.isna(low_pct_b) and low_pct_b >= 30:
        loc_hint_b = " low"
    if behind_pitches:
        bp_name, bp_usage = behind_pitches[0]
        # Check if we have an edge on this pitch
        bp_edge = pitch_details.get(bp_name, {}).get("edge", "")
        if bp_edge == "Advantage":
            behind_note = f"Expect **{bp_name}{loc_hint_b}** ({bp_usage:.0f}%) \u2014 you handle it, look to drive"
        elif bp_edge == "Vulnerable":
            behind_note = f"Expect **{bp_name}{loc_hint_b}** ({bp_usage:.0f}%) \u2014 you struggle here, protect zone"
        else:
            behind_note = f"Expect **{bp_name}{loc_hint_b}** ({bp_usage:.0f}%) \u2014 protect the zone, don't expand"
    elif hard_pitches:
        behind_note = f"Likely stays with **{hard_pitches[0][0]}** \u2014 shorten up, put it in play"

    # Count EV behind check
    count_behind_note = ""
    for ck_b in ["0-1", "1-2", "0-2"]:
        cd_b = count_ev.get(ck_b, {})
        cev_b = cd_b.get("avg_ev", np.nan)
        cn_b = cd_b.get("n", 0)
        overall_ev_b = hitter_profile.get("overall", {}).get("avg_ev", np.nan)
        if not pd.isna(cev_b) and not pd.isna(overall_ev_b) and cn_b >= 5:
            if cev_b >= 88:
                count_behind_note = f"You still hit well behind ({ck_b}: {cev_b:.0f} EV) \u2014 don't give away ABs"
                break
            elif (overall_ev_b - cev_b) > 8:
                count_behind_note = f"EV drops to {cev_b:.0f} when behind ({ck_b}) \u2014 be selectively aggressive earlier"
                break

    when_behind = {
        "note": behind_note,
        "count_note": count_behind_note,
    }

    # ── Section 4: TWO STRIKES ──
    putaway_cands = pitcher_profile.get("putaway_candidates", {})
    if putaway_cands:
        putaway_pitch = max(putaway_cands, key=putaway_cands.get)
        putaway_usage = putaway_cands[putaway_pitch]
    elif os_pitches:
        putaway_pitch, putaway_usage = os_pitches[0]
    elif hard_pitches:
        putaway_pitch, putaway_usage = hard_pitches[0]
    elif sorted_mix:
        putaway_pitch, putaway_usage = sorted_mix[-1] if len(sorted_mix) > 1 else sorted_mix[0]
    else:
        putaway_pitch, putaway_usage = "?", 0

    # Putaway location
    putaway_is_hard = putaway_pitch in _hard_set
    low_pct = pitcher_profile.get("loc_low_pct", np.nan)
    if not pd.isna(low_pct) and low_pct > 35:
        ts_loc = "low"
    elif putaway_is_hard:
        ts_loc = "up in zone"
    else:
        ts_loc = "below zone"

    ts_expect = f"{putaway_pitch} {ts_loc} ({putaway_usage:.0f}%)"

    # Chase vulnerability: check ALL 4 zones
    chase_vulns = _chase_vulnerability(zones)
    chase_warning = ""
    if chase_vulns:
        worst_cz = chase_vulns[0]
        cz_label = {"in": "inside", "away": "outside", "up": "above", "down": "below"}.get(worst_cz[0], worst_cz[0])
        chase_warning = f"\u26a0 DON'T CHASE \u2014 {worst_cz[1]:.0f}% whiff {cz_label} zone"

    # Protect plan — check BOTH classes' whiff to find true vulnerability
    hard_whiff_2k = by_class.get("hard", {}).get("whiff_pct", np.nan)
    os_whiff_2k = by_class.get("offspeed", {}).get("whiff_pct", np.nan)
    hard_adv = any(pe["edge"] == "Advantage" and pe["pitch"] in _hard_set for pe in pitch_edges)
    # Determine which class is the REAL vulnerability (regardless of putaway pitch)
    hard_vuln = not pd.isna(hard_whiff_2k) and hard_whiff_2k > 30
    os_vuln = not pd.isna(os_whiff_2k) and os_whiff_2k > 30
    if not pd.isna(chase) and chase < 22 and not pd.isna(z_contact) and z_contact > 80:
        protect = "Battle in zone, you're tough to K"
    elif not pd.isna(chase) and chase > 28:
        protect = "Shorten up, don't expand"
    elif hard_vuln and not os_vuln:
        # Vulnerable to hard stuff — gear for fastball
        if hard_adv:
            protect = f"Hard whiff is high ({hard_whiff_2k:.0f}%) but you crush it \u2014 time it up, don't be late"
        else:
            protect = f"Cheat fastball ({hard_whiff_2k:.0f}% 2K whiff), react to offspeed"
    elif os_vuln and not hard_vuln:
        # Vulnerable to offspeed — sit offspeed or cheat hard
        if hard_adv:
            protect = f"Cheat hard (Advantage), spit offspeed out of zone ({os_whiff_2k:.0f}% 2K whiff)"
        else:
            protect = f"Look for offspeed ({os_whiff_2k:.0f}% 2K whiff), fight off hard"
    elif hard_vuln and os_vuln:
        protect = "High whiff both classes \u2014 shorten up, protect zone, foul off tough pitches"
    else:
        protect = f"Shorten up, fight off {putaway_pitch}"

    # 3C: Count-aware EV callout — 2 STRIKES
    count_2k_note = ""
    overall_ev = hitter_profile.get("overall", {}).get("avg_ev", np.nan)
    for ck2 in ["0-2", "1-2"]:
        cd2 = count_ev.get(ck2, {})
        cev2 = cd2.get("avg_ev", np.nan)
        if not pd.isna(cev2) and not pd.isna(overall_ev):
            if cev2 < 80 or (overall_ev - cev2) > 8:
                count_2k_note = f"EV drops to {cev2:.0f} with 2 strikes \u2014 get your A-swing early"
                break

    # 3D: 2K pitch-class whiff note
    two_k_class_note = ""
    if not pd.isna(two_k_whiff_hard) and not pd.isna(two_k_whiff_os):
        diff = two_k_whiff_os - two_k_whiff_hard
        if abs(diff) >= 12:
            if diff > 0:
                two_k_class_note = (f"Your OS whiff with 2 strikes is {two_k_whiff_os:.0f}% "
                                    f"but hard whiff is only {two_k_whiff_hard:.0f}% "
                                    f"\u2014 look offspeed, react hard")
            else:
                two_k_class_note = (f"Your hard whiff with 2 strikes is {two_k_whiff_hard:.0f}% "
                                    f"but OS whiff is only {two_k_whiff_os:.0f}% "
                                    f"\u2014 cheat fastball, react offspeed")

    # Putaway danger ranking — rank pitcher's pitches by how dangerous they are to us with 2K
    putaway_danger = []
    for p, u in sorted_mix:
        if u < 5:
            continue
        is_hard_p = p in _hard_set
        our_2k_wh = two_k_whiff_hard if is_hard_p else two_k_whiff_os
        their_whiff_p = pitcher_profile.get("pitch_whiff", {}).get(p, np.nan)
        their_chase_p = pitcher_profile.get("pitch_chase", {}).get(p, np.nan)
        # Score: how dangerous is this pitch to us on 2 strikes?
        # Weight: our 2K vulnerability (50%) + their pitch whiff (25%) + their chase gen (25%)
        t2k = our_2k_wh / 100 if not pd.isna(our_2k_wh) else 0.25
        tw = their_whiff_p / 100 if not pd.isna(their_whiff_p) else 0.15
        tc = their_chase_p / 100 if not pd.isna(their_chase_p) else 0.10
        danger = t2k * 0.50 + tw * 0.25 + tc * 0.25
        putaway_danger.append((p, round(danger * 100, 1), our_2k_wh, is_hard_p, their_whiff_p, u))
    putaway_danger.sort(key=lambda x: x[1], reverse=True)

    two_strike = {
        "expect": ts_expect,
        "chase_warning": chase_warning,
        "protect": protect,
        "count_note": count_2k_note,
        "class_whiff_note": two_k_class_note,
        "putaway_danger": putaway_danger[:3],  # top 3 most dangerous pitches
    }

    # ── 3E: Pull tendency warning ──
    pull_warning = ""
    pull_pct = spray.get("pull_pct", 0)
    loc_outside = pitcher_profile.get("loc_outside_pct", np.nan)
    if pull_pct > 55 and not pd.isna(loc_outside) and loc_outside >= 30:
        pull_warning = f"You pull {pull_pct:.0f}% \u2014 pitcher lives outside ({loc_outside:.0f}%), stay through it."

    # ── Section 5: PITCH EDGES ──
    edge_lines = []
    sorted_edges = sorted(pitch_edges,
                          key=lambda pe: {"Advantage": 0, "Neutral": 1, "Vulnerable": 2}[pe["edge"]])
    for pe in sorted_edges:
        if pe["usage"] < 5:
            continue
        sym = "\u2713" if pe["edge"] == "Advantage" else ("\u2717" if pe["edge"] == "Vulnerable" else "\u2014")
        ev_str = f"{pe['our_ev']:.0f} EV" if not pd.isna(pe["our_ev"]) else "no data"
        wh_str = f"{pe['our_whiff']:.0f}% whiff" if not pd.isna(pe["our_whiff"]) else ""
        reason_short = pe.get("reason", "")
        # Add timing note for vulnerable pitches with depth info
        tn = pe.get("timing_note", "")
        if tn and pe["edge"] == "Vulnerable":
            reason_short += f" ({tn})"
        parts = [ev_str]
        if wh_str:
            parts.append(wh_str)
        stats_str = ", ".join(parts)
        edge_lines.append({
            "symbol": sym,
            "pitch": pe["pitch"],
            "edge": pe["edge"],
            "stats": stats_str,
            "reason": reason_short,
        })

    # ── TAG LINE data ──
    tag_line_data = {
        "name": hitter_profile["name"],
        "hand": hitter_profile["bats"],
        "swing_type": swing_type,
        "bat_speed": f"{bs_avg:.0f}" if bs_avg is not None else "-",
        "discipline_grade": disc_grade,
        "chase_pct": f"{chase:.0f}" if not pd.isna(chase) else "-",
        "tag_line": tag_line,
    }

    return {
        "tag_line": tag_line_data,
        "scout_lines": scout_lines,
        "first_pitch": first_pitch,
        "when_ahead": when_ahead,
        "when_behind": when_behind,
        "two_strike": two_strike,
        "pitch_edges": edge_lines,
        "pull_warning": pull_warning,
    }


# ── Game Plan UI ──

def _game_plan_tab(tm, team, data):
    """Game Plan tab — unified pitching + hitting plans."""
    section_header(f"Game Plan vs {team}")
    st.caption("Cross-referencing TrueMedia season data with our Trackman pitch-level data")
    seasons = get_all_seasons()
    season_filter = st.multiselect("Our Trackman Seasons", seasons, default=seasons, key="gpl_season")
    tab_pitch, tab_hit = st.tabs(["Our Pitching Plan", "Our Hitting Plan"])
    with tab_pitch:
        try:
            _pitching_plan_content(tm, team, data, season_filter)
        except Exception as e:
            st.error(f"Pitching plan error: {e}")
    with tab_hit:
        try:
            _hitting_plan_content(tm, team, data, season_filter)
        except Exception as e:
            st.error(f"Hitting plan error: {e}")


_ZONE_LABELS = {"up": "up in zone", "down": "down in zone", "glove": "glove side",
                 "arm": "arm side", "chase_low": "below zone"}

def _best_putaway_zone(pitch_name, ars_pt):
    """Find the best putaway location for a pitch using zone_eff + PZM.
    Returns (zone_label, whiff_pct) or (None, None) if no data."""
    ze = ars_pt.get("zone_eff", {})
    if not ze:
        return None, None
    ivb = ars_pt.get("ivb", np.nan)
    is_hard = pitch_name in _hard_pitches
    # Pitch-design zone multipliers (same as composite scorer)
    if is_hard:
        if pitch_name == "Sinker":
            pzm = {"up": 0.5, "down": 1.4, "chase_low": 1.3, "glove": 1.0, "arm": 1.0}
        elif not pd.isna(ivb) and ivb >= 16:
            pzm = {"up": 1.4, "down": 0.7, "chase_low": 0.4, "glove": 1.0, "arm": 1.0}
        elif not pd.isna(ivb) and ivb < 12:
            pzm = {"up": 0.7, "down": 1.3, "chase_low": 1.1, "glove": 1.0, "arm": 1.0}
        else:
            pzm = {"up": 1.15, "down": 0.9, "chase_low": 0.6, "glove": 1.0, "arm": 1.0}
    else:
        if pitch_name in ("Slider", "Sweeper"):
            pzm = {"up": 0.3, "down": 1.2, "chase_low": 1.4, "glove": 1.3, "arm": 0.8}
        elif pitch_name in ("Curveball", "Knuckle Curve"):
            pzm = {"up": 0.2, "down": 1.3, "chase_low": 1.5, "glove": 1.0, "arm": 1.0}
        elif pitch_name in ("Changeup", "Splitter"):
            pzm = {"up": 0.3, "down": 1.3, "chase_low": 1.3, "glove": 1.1, "arm": 0.9}
        else:
            pzm = {"up": 0.5, "down": 1.2, "chase_low": 1.3, "glove": 1.0, "arm": 1.0}
    # For putaway, heavily weight whiff_pct (we want swings and misses)
    best_zone, best_score = None, -1
    for zn, zdata in ze.items():
        w = zdata.get("whiff_pct", np.nan)
        n = zdata.get("n", 0)
        if pd.isna(w) or n < 5:
            continue
        # PZM-weighted whiff score
        score = w * pzm.get(zn, 1.0)
        if score > best_score:
            best_score = score
            best_zone = zn
    if best_zone is None:
        return None, None
    return _ZONE_LABELS.get(best_zone, best_zone), ze[best_zone].get("whiff_pct", np.nan)


def _pitching_plan_content(tm, team, data, season_filter):
    """Our Pitching Plan: how each of our pitchers should attack their lineup."""
    h_rate = _tm_team(tm["hitting"]["rate"], team)
    if h_rate.empty:
        st.info("No opponent hitting data found.")
        return
    dav_pitching = filter_davidson(data, role="pitcher")
    if season_filter:
        dav_pitching = dav_pitching[dav_pitching["Season"].isin(season_filter)]
    our_pitchers = sorted(dav_pitching["Pitcher"].unique())
    if not our_pitchers:
        st.warning("No Davidson pitcher data available.")
        return
    selected_pitcher = st.selectbox("Select Our Pitcher", our_pitchers,
                                     format_func=display_name, key="gpl_our_pitcher")
    tunnel_pop = build_tunnel_population_pop()
    arsenal = _get_our_pitcher_arsenal(data, selected_pitcher, season_filter, tunnel_pop=tunnel_pop)
    if arsenal is None:
        st.warning("Not enough Trackman data for this pitcher (min 100 pitches).")
        return
    throws_str = "RHP" if arsenal["throws"] == "Right" else "LHP"
    pitch_items = sorted(arsenal["pitches"].items(), key=lambda x: x[1]["usage_pct"], reverse=True)
    if not pitch_items:
        st.info("No pitch type breakdown available.")
        return
    # Prepare tunnel and sequence data for bullpen cards (no display here — see Pitching Lab)
    tunnels = arsenal.get("tunnels", pd.DataFrame())
    sequences = arsenal.get("sequences", pd.DataFrame())
    if isinstance(sequences, pd.DataFrame) and not sequences.empty:
        valid_pitches = {name for name, pt in arsenal["pitches"].items() if pt.get("count", 0) >= 10}
        sequences = sequences[
            sequences["Setup Pitch"].isin(valid_pitches) & sequences["Follow Pitch"].isin(valid_pitches)
        ]
        if "Count" in sequences.columns:
            sequences = sequences[sequences["Count"] >= 25]
    # ── Hitter-by-Hitter Plan ──
    opp_hitters = h_rate.sort_values("PA", ascending=False).head(12)
    pitch_summary = " / ".join(f"{n} {pt['usage_pct']:.0f}%" for n, pt in pitch_items[:4])
    section_header(f"{display_name(selected_pitcher)} ({throws_str}) vs Lineup")
    st.caption(f"Arsenal: {pitch_summary} — {arsenal['total_pitches']} pitches")
    all_matchups = []
    for _, row in opp_hitters.iterrows():
        hitter_name = row["playerFullName"]
        profile = _get_opp_hitter_profile(tm, hitter_name, team)
        matchup = _score_pitcher_vs_hitter(arsenal, profile)
        if matchup:
            all_matchups.append(matchup)
    if not all_matchups:
        st.info("No matchup data generated.")
        return
    all_matchups.sort(key=lambda x: x["overall_score"], reverse=True)
    # ── Helper for bullpen cards ──
    _hfmt = lambda v, fmt=".0f": f"{v:{fmt}}" if not pd.isna(v) else "-"

    # Pre-compute and cache sequences for each matchup (used by bullpen cards)
    for m in all_matchups:
        ps = m["pitch_scores"]
        hd = m.get("hitter_data", {})
        sorted_ps = sorted(ps.items(), key=lambda x: x[1]["score"], reverse=True)
        top_seqs = _build_3pitch_sequences(sorted_ps, hd, tunnels, sequences)
        m["_cached_seqs"] = top_seqs
    # ── Bullpen Cards ──
    st.markdown("---")
    section_header("Bullpen Cards")
    # D1-wide percentile series for all hitter tendency labels
    _all_sw = tm["hitting"].get("swing_stats", pd.DataFrame())
    _all_sp = tm["hitting"].get("swing_pct", pd.DataFrame())
    _throws_key = "RHP" if arsenal["throws"] == "Right" else "LHP"
    def _pct_series(df, col):
        return pd.to_numeric(df[col], errors="coerce").dropna() if col in df.columns else pd.Series(dtype=float)
    # 2K whiff rates (hand-matched)
    _2k_hard_series = _pct_series(_all_sw, f"2K Whiff vs {_throws_key} Hard")
    _2k_os_series = _pct_series(_all_sw, f"2K Whiff vs {_throws_key} OS")
    # 1P swing% vs hard/CH
    _fp_hard_series = _pct_series(_all_sw, "1PSwing% vs Hard Empty")
    _fp_ch_series = _pct_series(_all_sw, "1PSwing% vs CH Empty")
    # Overall swing% by pitch class
    _sw_hard_series = _pct_series(_all_sp, "Swing% vs Hard")
    _sw_sl_series = _pct_series(_all_sp, "Swing% vs SL")
    _sw_cb_series = _pct_series(_all_sp, "Swing% vs CB")
    _sw_ch_series = _pct_series(_all_sp, "Swing% vs CH")
    # Map pitch class → percentile series
    _sw_pct_map = {"swing_vs_hard": _sw_hard_series, "swing_vs_sl": _sw_sl_series,
                   "swing_vs_cb": _sw_cb_series, "swing_vs_ch": _sw_ch_series}
    # Helper: compute percentile with safety
    def _pctile(series, val):
        if pd.isna(val) or len(series) <= 10:
            return np.nan
        return percentileofscore(series, val, kind="rank")
    # Helper: label a percentile
    def _pct_label(pct):
        if pct >= 85: return "elite"
        if pct >= 70: return "above avg"
        if pct >= 60: return "exploitable"
        if pct <= 15: return "elite low"
        if pct <= 30: return "below avg"
        return "avg"

    for matchup in all_matchups:
        hd = matchup.get("hitter_data", {})
        ps = matchup["pitch_scores"]
        sorted_ps = sorted(ps.items(), key=lambda x: x[1]["score"], reverse=True)
        # Scores already computed by unified _pitch_score_composite in _score_pitcher_vs_hitter
        composites = {pt_name: round(pt_data["score"], 0) for pt_name, pt_data in sorted_ps}
        score = matchup["overall_score"]
        # Use cached sequences from summary table (avoid recomputation)
        top_seqs = matchup.get("_cached_seqs") or _build_3pitch_sequences(sorted_ps, hd, tunnels, sequences)
        # Expander label: clean and simple
        with st.expander(
            f"{'🟢' if score > 60 else '🟡' if score > 48 else '🔴'} "
            f"{display_name(matchup['hitter'])} ({matchup['bats']}) — {matchup['platoon']} — "
            f"Score: {score:.0f}"
        ):
            # ── Shared setup for bullpen card sections ──
            our_throws = arsenal["throws"]  # "Right" or "Left"
            their_bats = matchup["bats"]  # "R", "L", "S"
            same_side = (our_throws == "Right" and their_bats in ("R", "Right")) or \
                        (our_throws == "Left" and their_bats in ("L", "Left"))
            # Percentiles for all hitter tendencies
            their_2k_hard = hd.get("whiff_2k_hard", np.nan)
            their_2k_os = hd.get("whiff_2k_os", np.nan)
            pct_2k_os = _pctile(_2k_os_series, their_2k_os)
            pct_2k_hard = _pctile(_2k_hard_series, their_2k_hard)

            # ── Shared data ──
            ps_dict = dict(sorted_ps)
            real_ps = [(n, d) for n, d in sorted_ps if d.get("usage", 0) >= 10]
            if not real_ps:
                real_ps = sorted_ps[:3]
            real_hard = [(n, d) for n, d in real_ps if n in _hard_pitches]
            real_os = [(n, d) for n, d in real_ps if n not in _hard_pitches]
            primary = max(real_ps, key=lambda x: x[1].get("usage", 0) or 0) if real_ps else None
            best_hard_p = real_hard[0] if real_hard else (real_ps[0] if real_ps else None)
            their_chase_val = hd.get("chase_pct", np.nan)

            # ── RESOLVE SEQUENCE FIRST so scouting can reference it ──
            fp_hard = hd.get("fp_swing_hard", np.nan)
            fp_pct = _pctile(_fp_hard_series, fp_hard)
            fp_context = ""
            if not pd.isna(fp_hard) and not pd.isna(fp_pct):
                if fp_pct <= 25:
                    fp_context = "passive"
                elif fp_pct >= 75:
                    fp_context = "aggressive"

            # Determine the active sequence (the one that drives the game plan)
            seq_p1, seq_p2, seq_p3 = None, None, None
            g12, g23 = "-", "-"
            active_seq = None
            if top_seqs:
                active_seq = top_seqs[0]
                seq_p1, seq_p2, seq_p3 = active_seq["p1"], active_seq["p2"], active_seq["p3"]
                _, g12 = _lookup_tunnel(seq_p1, seq_p2, tunnels)
                _, g23 = _lookup_tunnel(seq_p2, seq_p3, tunnels)
                # Aggressive-1P override: swap to offspeed-starting seq
                if fp_context == "aggressive" and seq_p1 in _hard_pitches and real_os:
                    os_seqs = [s for s in top_seqs if s["p1"] not in _hard_pitches]
                    if os_seqs:
                        active_seq = os_seqs[0]
                        seq_p1, seq_p2, seq_p3 = active_seq["p1"], active_seq["p2"], active_seq["p3"]
                        _, g12 = _lookup_tunnel(seq_p1, seq_p2, tunnels)
                        _, g23 = _lookup_tunnel(seq_p2, seq_p3, tunnels)
                # Passive-1P override: force hard-pitch-starting seq (attack zone)
                elif fp_context == "passive" and seq_p1 not in _hard_pitches and real_hard:
                    hard_seqs = [s for s in top_seqs if s["p1"] in _hard_pitches]
                    if hard_seqs:
                        active_seq = hard_seqs[0]
                        seq_p1, seq_p2, seq_p3 = active_seq["p1"], active_seq["p2"], active_seq["p3"]
                        _, g12 = _lookup_tunnel(seq_p1, seq_p2, tunnels)
                        _, g23 = _lookup_tunnel(seq_p2, seq_p3, tunnels)

            # ── UNIFIED AT-BAT SCRIPT (V3 — full game-prep depth) ──
            real_ps_names = {n for n, _ in real_ps}

            if active_seq:
                st.markdown("**At-Bat Script**")

                # ── Hitter profile snapshot ──
                prof_parts = []
                h_chase = hd.get("chase_pct", np.nan)
                h_kpct = hd.get("k_pct", np.nan)
                h_bbpct = hd.get("bb_pct", np.nan)
                h_swstrk = hd.get("swstrk_pct", np.nan)
                h_contact = hd.get("contact_pct", np.nan)
                if not pd.isna(h_kpct):
                    prof_parts.append(f"K% {h_kpct:.0f}")
                if not pd.isna(h_bbpct):
                    prof_parts.append(f"BB% {h_bbpct:.0f}")
                if not pd.isna(h_chase):
                    prof_parts.append(f"Chase {h_chase:.0f}%")
                if not pd.isna(h_swstrk):
                    prof_parts.append(f"SwStr {h_swstrk:.1f}%")
                if not pd.isna(h_contact):
                    prof_parts.append(f"Contact {h_contact:.0f}%")
                # Zone tendencies
                h_high = hd.get("high_pct", np.nan)
                h_low = hd.get("low_pct", np.nan)
                h_in = hd.get("inside_pct", np.nan)
                h_out = hd.get("outside_pct", np.nan)
                zone_parts = []
                if not pd.isna(h_high) and h_high > 55:
                    zone_parts.append("high-ball hitter")
                elif not pd.isna(h_low) and h_low > 55:
                    zone_parts.append("low-ball hitter")
                if not pd.isna(h_in) and h_in > 55:
                    zone_parts.append("pulls inside")
                elif not pd.isna(h_out) and h_out > 55:
                    zone_parts.append("covers outside")
                if prof_parts:
                    zone_tag = f" — {', '.join(zone_parts)}" if zone_parts else ""
                    st.caption(f"{' | '.join(prof_parts)}{zone_tag}")

                # ── ① FIRST PITCH ──
                fp_ars = arsenal["pitches"].get(seq_p1, {})
                fp_loc, fp_whiff = _best_putaway_zone(seq_p1, fp_ars)
                fp_loc_str = f" ({fp_loc}, {fp_whiff:.0f}% whiff)" if fp_loc and not pd.isna(fp_whiff) else ""
                if fp_context == "passive":
                    st.markdown(f"**① FIRST PITCH:** {seq_p1} — attack zone{fp_loc_str}")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;_Passive 1P hitter ({fp_hard:.0f}% swing, {fp_pct:.0f}th %ile) — free strike_")
                elif fp_context == "aggressive":
                    st.markdown(f"**① FIRST PITCH:** {seq_p1} — steal strike{fp_loc_str}")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;_Aggressive 1P hitter ({fp_hard:.0f}% swing, {fp_pct:.0f}th %ile) — will chase_")
                else:
                    st.markdown(f"**① FIRST PITCH:** {seq_p1} — compete{fp_loc_str}")
                # Per-class swing tendencies on 1P
                fp_sw_parts = []
                for sw_label, sw_key in [("Hard", "swing_vs_hard"), ("SL", "swing_vs_sl"),
                                          ("CB", "swing_vs_cb"), ("CH", "swing_vs_ch")]:
                    sv = hd.get(sw_key, np.nan)
                    if not pd.isna(sv):
                        _sw_ser = _sw_pct_map.get(sw_key, pd.Series(dtype=float))
                        sv_pct = _pctile(_sw_ser, sv)
                        if not pd.isna(sv_pct) and (sv_pct >= 70 or sv_pct <= 25):
                            tag = "agg" if sv_pct >= 70 else "passive"
                            fp_sw_parts.append(f"{sw_label} {sv:.0f}% ({tag})")
                if fp_sw_parts:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;_Swing rates: {', '.join(fp_sw_parts)}_")

                # ── ② TOP SEQUENCES (hitter-aware 3-pitch paths) ──
                if top_seqs:
                    st.markdown("**② TOP SEQUENCES** _(hitter-aware)_")
                    for i, s in enumerate(top_seqs):
                        sw23 = s.get("sw23", np.nan)
                        t12 = s.get("t12", np.nan)
                        t23 = s.get("t23", np.nan)
                        ev_gap = s.get("effv_gap", np.nan)
                        t2k = s.get("their_2k", np.nan)
                        # Tunnel grades from lookup
                        _, g12_s = _lookup_tunnel(s["p1"], s["p2"], tunnels)
                        _, g23_s = _lookup_tunnel(s["p2"], s["p3"], tunnels)
                        # Build detail line
                        detail_parts = []
                        if g12_s not in ("-", "F", None):
                            detail_parts.append(f"P1→P2 {g12_s} tunnel")
                        if g23_s not in ("-", "F", None):
                            detail_parts.append(f"P2→P3 {g23_s} tunnel")
                        if not pd.isna(sw23):
                            detail_parts.append(f"{sw23:.0f}% whiff on P3")
                        if not pd.isna(ev_gap):
                            detail_parts.append(f"{ev_gap:.1f} mph EffV gap")
                        if not pd.isna(t2k):
                            is_hard_p3 = s["p3"] in _hard_pitches
                            detail_parts.append(f"hitter {t2k:.0f}% 2K {'hard' if is_hard_p3 else 'OS'}")
                        # Best location for putaway
                        p3_ars = arsenal["pitches"].get(s["p3"], {})
                        p3_loc, p3_whiff = _best_putaway_zone(s["p3"], p3_ars)
                        if p3_loc:
                            detail_parts.append(f"P3 loc: {p3_loc}")
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**{i+1}. {s['p1']} → {s['p2']} → {s['p3']}** (Score: {s['score']:.0f})")
                        if detail_parts:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_{' | '.join(detail_parts)}_")

                # ── 4-PITCH SEQUENCE (extended at-bat) ──
                seq4 = _build_4pitch_sequence(top_seqs, sorted_ps, hd, tunnels, sequences)
                if seq4:
                    detail4 = []
                    _, g12_4 = _lookup_tunnel(seq4["p1"], seq4["p2"], tunnels)
                    _, g23_4 = _lookup_tunnel(seq4["p2"], seq4["p3"], tunnels)
                    _, g34_4 = _lookup_tunnel(seq4["p3"], seq4["p4"], tunnels)
                    for label, grade in [("P1→P2", g12_4), ("P2→P3", g23_4), ("P3→P4", g34_4)]:
                        if grade not in ("-", "F", None):
                            detail4.append(f"{label} {grade}")
                    sw34 = seq4.get("sw34", np.nan)
                    if not pd.isna(sw34):
                        detail4.append(f"{sw34:.0f}% whiff on P4")
                    p4_ars = arsenal["pitches"].get(seq4["p4"], {})
                    p4_loc, _ = _best_putaway_zone(seq4["p4"], p4_ars)
                    if p4_loc:
                        detail4.append(f"P4 loc: {p4_loc}")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**4P. {seq4['seq']}** (Score: {seq4['score']:.0f})")
                    if detail4:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_{' | '.join(detail4)}_")

                # ── WHEN BEHIND (1-0, 2-0, 2-1, 3-1, 3-2) ──
                if best_hard_p:
                    bh_ars = arsenal["pitches"].get(best_hard_p[0], {})
                    bh_loc, bh_whiff = _best_putaway_zone(best_hard_p[0], bh_ars)
                    bh_loc_str = f" — {bh_loc}" if bh_loc else ""
                    bh_whiff_str = f" ({bh_whiff:.0f}% whiff)" if not pd.isna(bh_whiff) else ""
                    st.markdown(f"**WHEN BEHIND (1-0, 2-0, 3-2):** {best_hard_p[0]}{bh_loc_str}{bh_whiff_str}")
                    # Secondary behind option if we have a second hard pitch
                    if len(real_hard) >= 2:
                        bh2_name, bh2_d = real_hard[1]
                        bh2_ars = arsenal["pitches"].get(bh2_name, {})
                        bh2_loc, bh2_whiff = _best_putaway_zone(bh2_name, bh2_ars)
                        bh2_loc_str = f" — {bh2_loc}" if bh2_loc else ""
                        bh2_whiff_str = f" ({bh2_whiff:.0f}% whiff)" if not pd.isna(bh2_whiff) else ""
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;_Alt: {bh2_name}{bh2_loc_str}{bh2_whiff_str}_")

                # ── 2K APPROACH ──
                os_whiff_str = f"{their_2k_os:.0f}% OS whiff ({pct_2k_os:.0f}th %ile, {_pct_label(pct_2k_os)})" if not pd.isna(their_2k_os) and not pd.isna(pct_2k_os) else None
                hard_whiff_str = f"{their_2k_hard:.0f}% Hard whiff ({pct_2k_hard:.0f}th %ile, {_pct_label(pct_2k_hard)})" if not pd.isna(their_2k_hard) and not pd.isna(pct_2k_hard) else None
                twok_lines = []
                if os_whiff_str:
                    if not pd.isna(pct_2k_os) and pct_2k_os >= 60:
                        os_recs = []
                        for n, d in real_os:
                            ars_n = arsenal["pitches"].get(n, {})
                            loc, loc_w = _best_putaway_zone(n, ars_n)
                            loc_detail = f" {loc} ({loc_w:.0f}%)" if loc and not pd.isna(loc_w) else ""
                            os_recs.append(f"{n}{loc_detail}")
                        twok_lines.append(f"{os_whiff_str} → {' / '.join(os_recs)}" if os_recs else os_whiff_str)
                    elif not pd.isna(pct_2k_os) and pct_2k_os <= 30:
                        twok_lines.append(f"{os_whiff_str} → avoid offspeed putaway")
                    else:
                        twok_lines.append(os_whiff_str)
                if hard_whiff_str:
                    if not pd.isna(pct_2k_hard) and pct_2k_hard >= 60:
                        hard_recs = []
                        for n, d in real_hard:
                            ars_n = arsenal["pitches"].get(n, {})
                            loc, loc_w = _best_putaway_zone(n, ars_n)
                            loc_detail = f" {loc} ({loc_w:.0f}%)" if loc and not pd.isna(loc_w) else ""
                            hard_recs.append(f"{n}{loc_detail}")
                        twok_lines.append(f"{hard_whiff_str} → {' / '.join(hard_recs)}" if hard_recs else hard_whiff_str)
                    elif not pd.isna(pct_2k_hard) and pct_2k_hard <= 30:
                        twok_lines.append(f"{hard_whiff_str} → avoid fastball putaway")
                    else:
                        twok_lines.append(hard_whiff_str)
                if twok_lines:
                    st.markdown("**2K APPROACH:**")
                    for line in twok_lines:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{line}")
                    # Per-pitch putaway ranking for 2K counts
                    # Hitter's 2K whiff by class is dominant — it's the hitter's
                    # actual vulnerability, not our pitcher's general whiff rate.
                    pw_rank = []
                    for n, d in real_ps:
                        pw_w = d.get("our_whiff", 0) or 0
                        pw_ch = d.get("our_chase", 0) or 0
                        is_h = n in _hard_pitches
                        t2k_val = hd.get("whiff_2k_hard" if is_h else "whiff_2k_os", np.nan)
                        ars_n = arsenal["pitches"].get(n, {})
                        loc, _ = _best_putaway_zone(n, ars_n)
                        # Weight: hitter 2K whiff (50%) > our whiff (25%) > chase (25%)
                        pw_score = (t2k_val * 0.50 if not pd.isna(t2k_val) else 0) + pw_w * 0.25 + pw_ch * 0.25
                        pw_rank.append((n, pw_score, pw_w, pw_ch, loc))
                    pw_rank.sort(key=lambda x: x[1], reverse=True)
                    if pw_rank:
                        st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;_Putaway ranking:_")
                        for rk_name, rk_sc, rk_w, rk_ch, rk_loc in pw_rank[:4]:
                            loc_tag = f" → {rk_loc}" if rk_loc else ""
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_{rk_name}: {rk_w:.0f}% whiff, {rk_ch:.0f}% chase{loc_tag}_")

                # ── Compact context footer ──
                footer_parts = []
                if not pd.isna(their_chase_val):
                    if their_chase_val > 32:
                        best_chase_p = max(real_ps, key=lambda x: x[1].get("our_chase", 0) or 0)
                        footer_parts.append(f"Chaser ({their_chase_val:.0f}% chase) — expand early with {best_chase_p[0]}")
                    elif their_chase_val < 18:
                        footer_parts.append(f"Elite discipline ({their_chase_val:.0f}% chase) — must throw strikes")
                if not footer_parts and not pd.isna(hd.get("bb_pct", np.nan)) and hd["bb_pct"] > 14:
                    footer_parts.append(f"Patient ({hd['bb_pct']:.0f}% BB) — attack zone")
                if footer_parts:
                    st.caption(" | ".join(footer_parts))

            else:
                # Fallback: no sequences — show all available data
                st.markdown("**At-Bat Script**")
                p1_name = primary[0] if primary else (best_hard_p[0] if best_hard_p else None)
                if p1_name:
                    p1_ars = arsenal["pitches"].get(p1_name, {})
                    p1_loc, p1_whiff = _best_putaway_zone(p1_name, p1_ars)
                    p1_loc_str = f" ({p1_loc}, {p1_whiff:.0f}% whiff)" if p1_loc and not pd.isna(p1_whiff) else ""
                    st.markdown(f"**① FIRST PITCH:** {p1_name} — compete{p1_loc_str}")
                if best_hard_p:
                    bh_ars = arsenal["pitches"].get(best_hard_p[0], {})
                    bh_loc, bh_whiff = _best_putaway_zone(best_hard_p[0], bh_ars)
                    bh_loc_str = f" — {bh_loc}" if bh_loc else ""
                    bh_whiff_str = f" ({bh_whiff:.0f}% whiff)" if not pd.isna(bh_whiff) else ""
                    st.markdown(f"**WHEN BEHIND (1-0, 2-0, 3-2):** {best_hard_p[0]}{bh_loc_str}{bh_whiff_str}")


def _hitting_plan_content(tm, team, data, season_filter):
    """Our Hitting Plan V2: pitcher scouting report + actionable hitter cards."""
    p_trad = _tm_team(tm["pitching"]["traditional"], team)
    if p_trad.empty:
        st.info("No opponent pitching data found.")
        return
    pitcher_ip = {}
    for name in p_trad["playerFullName"].unique():
        ip = _safe_num(_tm_player(p_trad, name), "IP")
        pitcher_ip[name] = ip if not pd.isna(ip) else 0
    opp_pitchers = sorted(pitcher_ip.keys(), key=lambda x: pitcher_ip[x], reverse=True)
    selected_opp = st.selectbox("Select Their Pitcher", opp_pitchers, key="gpl_opp_pitcher")
    opp_profile = _get_opp_pitcher_profile(tm, selected_opp, team)
    throws_str = "RHP" if opp_profile["throws"] == "R" else ("LHP" if opp_profile["throws"] == "L" else "BHP")
    st.markdown(f"### vs {selected_opp} ({throws_str})")

    # ── A. Pitcher Scouting Report ──
    col_ars, col_stats = st.columns([3, 1])
    with col_ars:
        if opp_profile["pitch_mix"]:
            arsenal_rows = sorted(opp_profile["pitch_mix"].items(), key=lambda x: x[1], reverse=True)
            fig = go.Figure(go.Bar(
                x=[r[0] for r in arsenal_rows],
                y=[r[1] for r in arsenal_rows],
                marker_color=[PITCH_COLORS.get(r[0], "#888") for r in arsenal_rows],
                text=[f"{r[1]:.1f}%" for r in arsenal_rows], textposition="outside",
                textfont=dict(size=11, color="#1a1a2e"),
            ))
            fig.update_layout(**CHART_LAYOUT, height=250, yaxis_title="Usage %", showlegend=False,
                              yaxis=dict(range=[0, max(r[1] for r in arsenal_rows) * 1.3]))
            st.plotly_chart(fig, use_container_width=True)
    with col_stats:
        c1, c2 = st.columns(2)
        with c1:
            st.metric("ERA", f"{opp_profile['era']:.2f}" if not pd.isna(opp_profile["era"]) else "?")
            st.metric("Velo", f"{opp_profile['velo']:.1f}" if not pd.isna(opp_profile["velo"]) else "?")
        with c2:
            st.metric("SwStr%", f"{opp_profile['swstrk_pct']:.1f}" if not pd.isna(opp_profile["swstrk_pct"]) else "?")
            st.metric("Chase%", f"{opp_profile['chase_pct']:.1f}" if not pd.isna(opp_profile["chase_pct"]) else "?")

    # "How to Attack" summary
    st.markdown("**How to Attack**")
    attack_notes = []
    loc_tend = opp_profile.get("location_tendency", "")
    h_pct = opp_profile.get("loc_high_pct", np.nan)
    l_pct = opp_profile.get("loc_low_pct", np.nan)
    # Location tendency — pick dominant direction, don't show both up AND down
    has_up = not pd.isna(h_pct) and h_pct >= 30
    has_down = not pd.isna(l_pct) and l_pct >= 35
    if has_up and has_down:
        # Both high and low — pick the stronger tendency
        if h_pct >= l_pct:
            attack_notes.append(f"Lives up in zone ({h_pct:.0f}%) \u2014 sit on elevated FB if you handle high heat")
        else:
            attack_notes.append(f"Works down ({l_pct:.0f}%) \u2014 look low, don't chase below zone")
    elif has_up:
        attack_notes.append(f"Lives up in zone ({h_pct:.0f}%) \u2014 sit on elevated FB if you handle high heat")
    elif has_down:
        attack_notes.append(f"Works down ({l_pct:.0f}%) \u2014 look low, don't chase below zone")
    comploc = opp_profile.get("comploc_pct", np.nan)
    if not pd.isna(comploc) and comploc < 40:
        attack_notes.append(f"Low command ({comploc:.0f}% CompLoc) \u2014 expect hittable mistakes")
    # Tempo advice — don't say both "be aggressive early" AND "work counts"
    low_swstr = not pd.isna(opp_profile["swstrk_pct"]) and opp_profile["swstrk_pct"] < 8
    high_bb9 = not pd.isna(opp_profile["bb9"]) and opp_profile["bb9"] > 4
    if low_swstr and high_bb9:
        attack_notes.append(f"Low whiff ({opp_profile['swstrk_pct']:.1f}% SwStr) + walks ({opp_profile['bb9']:.1f} BB/9) \u2014 hunt early, make him throw strikes")
    elif low_swstr:
        attack_notes.append(f"Low swing-and-miss ({opp_profile['swstrk_pct']:.1f}% SwStr) \u2014 be aggressive early")
    elif high_bb9:
        attack_notes.append(f"Control issues ({opp_profile['bb9']:.1f} BB/9) \u2014 work counts")
    if not pd.isna(opp_profile["ev_against"]) and opp_profile["ev_against"] > 88:
        attack_notes.append(f"Gives up hard contact ({opp_profile['ev_against']:.1f} mph EV) \u2014 look to drive")
    # Putaway pitch warning
    putaway_cands = opp_profile.get("putaway_candidates", {})
    opp_swstr = opp_profile.get("swstrk_pct", np.nan)
    if putaway_cands:
        top_putaway = max(putaway_cands, key=putaway_cands.get)
        if not pd.isna(opp_swstr) and opp_swstr > 10:
            attack_notes.append(f"{top_putaway} is putaway ({putaway_cands[top_putaway]:.0f}%) \u2014 recognize early, don't chase low")
    p_per_bf = opp_profile.get("p_per_bf", np.nan)
    if not pd.isna(p_per_bf) and p_per_bf > 4.0:
        attack_notes.append(f"Avg {p_per_bf:.1f} P/BF \u2014 be ready for deep counts")
    if opp_profile["pitch_mix"]:
        primary_p = max(opp_profile["pitch_mix"].items(), key=lambda x: x[1])
        if primary_p[1] > 55:
            attack_notes.append(f"Relies heavily on {primary_p[0]} ({primary_p[1]:.0f}%) \u2014 sit on it early")
    for note in attack_notes[:4]:
        st.markdown(f"- {note}")
    st.markdown("---")

    # ── Score all hitters ──
    dav_hitting = filter_davidson(data, role="batter")
    if season_filter:
        dav_hitting = dav_hitting[dav_hitting["Season"].isin(season_filter)]
    our_hitters = sorted(h for h in dav_hitting["Batter"].unique() if _is_position_player(h))
    all_matchups = []
    hitter_profiles = {}
    for hitter_name in our_hitters:
        hitter_tm = _get_our_hitter_profile(data, hitter_name, season_filter)
        if hitter_tm is None:
            continue
        matchup = _score_hitter_vs_pitcher(hitter_tm, opp_profile)
        if matchup:
            hitter_profiles[hitter_name] = hitter_tm
            script = _generate_at_bat_script(hitter_tm, opp_profile, matchup)
            matchup["script"] = script
            all_matchups.append(matchup)
    if not all_matchups:
        st.warning("No matchup data available.")
        return
    all_matchups.sort(key=lambda x: x["overall_score"], reverse=True)

    # ── B. Lineup Rankings ──
    section_header("Lineup Rankings")
    summary_rows = []
    for i, m in enumerate(all_matchups):
        script = m.get("script")
        tag = script.get("tag_line", {}) if script else {}
        # Best pitch (Advantage)
        adv_pitches = [pe for pe in m.get("pitch_edges", []) if pe["edge"] == "Advantage"]
        vuln_pitches = [pe for pe in m.get("pitch_edges", []) if pe["edge"] == "Vulnerable"]
        best_str = "-"
        if adv_pitches:
            bp = max(adv_pitches, key=lambda pe: pe["our_ev"] if not pd.isna(pe["our_ev"]) else 0)
            ev_s = f"{bp['our_ev']:.0f} EV" if not pd.isna(bp["our_ev"]) else ""
            best_str = f"{bp['pitch']} ({ev_s})"
        watch_str = "-"
        if vuln_pitches:
            wp = max(vuln_pitches, key=lambda pe: pe["our_whiff"] if not pd.isna(pe["our_whiff"]) else 0)
            wh_s = f"{wp['our_whiff']:.0f}% whiff" if not pd.isna(wp["our_whiff"]) else ""
            watch_str = f"{wp['pitch']} ({wh_s})"
        # Key stat: pick most distinctive attribute
        disc_g = tag.get("discipline_grade", "-")
        bs_val = tag.get("bat_speed", "-")
        key_stat = f"{disc_g} disc" if disc_g not in ("-", "C", "D") else (f"{bs_val} bat" if bs_val != "-" else "-")
        # Overall EV from profile
        h_prof = hitter_profiles.get(m["hitter"])
        ov_ev = h_prof.get("overall", {}).get("avg_ev", np.nan) if h_prof else np.nan
        ov_ev_str = f"{ov_ev:.0f}" if not pd.isna(ov_ev) else "-"
        summary_rows.append({
            "Rank": i + 1,
            "Hitter": display_name(m["hitter"]),
            "Pos": POSITION.get(m["hitter"], ""),
            "Bats": m["bats"],
            "Platoon": m["platoon"],
            "Edge": f"{m['overall_score']:.0f}",
            "Avg EV": ov_ev_str,
            "Best Pitch": best_str,
            "Watch Out": watch_str,
            "Key Stat": key_stat,
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # ── C. Top 9 Cards ──
    section_header("Recommended Top 9")
    top9 = all_matchups[:9]
    for row_start in range(0, len(top9), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx >= len(top9):
                break
            m = top9[idx]
            with col:
                clr = "#2ca02c" if m["overall_score"] > 60 else "#f7c631" if m["overall_score"] > 48 else "#d22d49"
                script = m.get("script")
                tag = script.get("tag_line", {}) if script else {}
                # One-line matchup summary
                adv_p = [pe for pe in m.get("pitch_edges", []) if pe["edge"] == "Advantage"]
                vuln_p = [pe for pe in m.get("pitch_edges", []) if pe["edge"] == "Vulnerable"]
                parts = []
                if adv_p:
                    bp = max(adv_p, key=lambda pe: pe["our_ev"] if not pd.isna(pe["our_ev"]) else 0)
                    ev_tag = f" ({bp['our_ev']:.0f})" if not pd.isna(bp["our_ev"]) else ""
                    parts.append(f"Sit {bp['pitch']}{ev_tag}")
                if vuln_p:
                    wp = max(vuln_p, key=lambda pe: pe["our_whiff"] if not pd.isna(pe["our_whiff"]) else 0)
                    wh_tag = f" ({wp['our_whiff']:.0f}%wh)" if not pd.isna(wp["our_whiff"]) else ""
                    parts.append(f"protect {wp['pitch']}{wh_tag}")
                summary_line = ", ".join(parts) if parts else "Neutral matchup"
                # Tag line info for card
                tag_str = tag.get("tag_line", "")
                # Position
                pos_str = POSITION.get(m["hitter"], "")
                # Barrel hunt one-liner for card
                barrel_line = ""
                if script:
                    bh = script.get("when_ahead", {}).get("barrel_hunt", "")
                    if bh:
                        barrel_line = bh.split(".")[0]  # first sentence only
                st.markdown(
                    f'<div style="padding:10px;background:white;border-radius:8px;border:1px solid #eee;'
                    f'border-left:4px solid {clr};margin:4px 0;">'
                    f'<div style="font-size:18px;font-weight:800;color:#1a1a2e;">#{idx+1}</div>'
                    f'<div style="font-size:14px;font-weight:700;color:#1a1a2e;">'
                    f'{display_name(m["hitter"])}</div>'
                    f'<div style="font-size:11px;color:#666;">{pos_str} | {m["platoon"]} | '
                    f'Edge: {m["overall_score"]:.0f}</div>'
                    f'<div style="font-size:10px;color:#888;margin-top:2px;">{tag_str}</div>'
                    f'<div style="font-size:11px;color:#333;font-weight:600;margin-top:4px;">{summary_line}</div>'
                    + (f'<div style="font-size:10px;color:#1a7a3a;margin-top:2px;">{barrel_line}</div>'
                       if barrel_line else '')
                    + f'</div>', unsafe_allow_html=True)

    # ── D. Per-Hitter Expanders ──
    st.markdown("---")
    section_header("At-Bat Scripts")
    for m in all_matchups:
        script = m.get("script")
        with st.expander(
            f"{'🟢' if m['overall_score'] > 60 else '🟡' if m['overall_score'] > 48 else '🔴'} "
            f"{display_name(m['hitter'])} ({m['bats']}) \u2014 Edge: {m['overall_score']:.0f} | {m['platoon']}"
        ):
            if not script:
                st.caption("No Trackman data \u2014 use pitcher tendencies above.")
                continue

            tag = script.get("tag_line", {})
            scout_lines = script.get("scout_lines", [])

            # ── Tag Line ──
            st.markdown(f"**{display_name(tag.get('name', m['hitter']))}** \u2014 {tag.get('tag_line', '')}")

            # ── Hit Lab Profile (barrel zones, attack angle, sweet spot) ──
            h_prof = hitter_profiles.get(m["hitter"], {})
            lab_parts = []
            ov = h_prof.get("overall", {})
            hh_pct = ov.get("hard_hit_pct", np.nan)
            ss_pct = ov.get("sweet_spot_pct", np.nan)
            brl_ct = ov.get("barrel_count", 0)
            sp_data = h_prof.get("swing_path") or {}
            aa = sp_data.get("attack_angle")
            if not pd.isna(hh_pct):
                lab_parts.append(f"{hh_pct:.0f}% hard hit")
            if not pd.isna(ss_pct):
                lab_parts.append(f"{ss_pct:.0f}% sweet spot")
            if brl_ct and brl_ct >= 2:
                lab_parts.append(f"{brl_ct} barrels")
            if aa is not None:
                lab_parts.append(f"{aa:.1f}\u00b0 attack angle")
            bz = h_prof.get("barrel_zones", [])
            if bz:
                top_bz = bz[0]
                lab_parts.append(f"barrels {top_bz[0].replace('_', ' ').lower()} ({top_bz[1]} barrels, {top_bz[2]:.0f} EV)")
            if lab_parts:
                st.caption(" | ".join(lab_parts))

            # ── Scouting This Pitcher ──
            if scout_lines:
                st.markdown("**SCOUTING THIS PITCHER**")
                st.markdown(" | ".join(scout_lines))

            # ── At-Bat Sections ──
            c1, c2 = st.columns(2)
            fp = script.get("first_pitch", {})
            with c1:
                st.markdown("**\u2460 1ST PITCH**")
                st.markdown(f"Expect: **{fp.get('expect', '?')}**")
                st.markdown(f"Plan: **{fp.get('plan', '?')}** \u2014 {fp.get('detail', '')}")
                if fp.get("look"):
                    st.markdown(f"Look: {fp['look']}")

                st.markdown("")
                ts = script.get("two_strike", {})
                st.markdown("**\u2462 2 STRIKES**")
                st.markdown(f"He'll throw: **{ts.get('expect', '?')}**")
                if ts.get("chase_warning"):
                    st.markdown(f"**{ts['chase_warning']}**")
                st.markdown(f"Protect: {ts.get('protect', 'Shorten up')}")
                if ts.get("count_note"):
                    st.markdown(f"*{ts['count_note']}*")
                if ts.get("class_whiff_note"):
                    st.markdown(f"*{ts['class_whiff_note']}*")
                # Putaway danger ranking
                pw_danger = ts.get("putaway_danger", [])
                if pw_danger and len(pw_danger) >= 2:
                    st.markdown("**Watch for (2K):**")
                    for pw_tup in pw_danger:
                        pw_p, pw_score, pw_2k_wh, pw_is_hard = pw_tup[0], pw_tup[1], pw_tup[2], pw_tup[3]
                        pw_their_wh = pw_tup[4] if len(pw_tup) > 4 else np.nan
                        pw_usage = pw_tup[5] if len(pw_tup) > 5 else np.nan
                        parts = []
                        if not pd.isna(pw_usage):
                            parts.append(f"{pw_usage:.0f}% usage")
                        if not pd.isna(pw_their_wh):
                            parts.append(f"{pw_their_wh:.0f}% their whiff")
                        if not pd.isna(pw_2k_wh):
                            cls_lbl = "hard" if pw_is_hard else "OS"
                            parts.append(f"your {cls_lbl} 2K whiff {pw_2k_wh:.0f}%")
                        st.markdown(f"- **{pw_p}**: {', '.join(parts)}" if parts else f"- {pw_p}")

            ah = script.get("when_ahead", {})
            with c2:
                st.markdown("**\u2461 AHEAD IN COUNT**")
                st.markdown(f"{ah.get('sit_on', 'Be aggressive in zone')}")
                if ah.get("barrel_hunt"):
                    st.markdown(f"**{ah['barrel_hunt']}**")
                elif ah.get("hunt_zone"):
                    st.markdown(f"Hunt zone: {ah['hunt_zone']}")
                if ah.get("count_note"):
                    st.markdown(f"*{ah['count_note']}*")

                # ── When Behind ──
                wb = script.get("when_behind", {})
                if wb.get("note") or wb.get("count_note"):
                    st.markdown("")
                    st.markdown("**WHEN BEHIND**")
                    if wb.get("note"):
                        st.markdown(wb["note"])
                    if wb.get("count_note"):
                        st.markdown(f"*{wb['count_note']}*")

                st.markdown("")
                # ── Pitch Edges ──
                edges = script.get("pitch_edges", [])
                if edges:
                    st.markdown("**PITCH EDGES**")
                    for el in edges:
                        sym = el["symbol"]
                        edge_lbl = el["edge"]
                        line = f"{sym} **{el['pitch']}** \u2014 {edge_lbl}: {el['stats']}"
                        if el.get("reason") and edge_lbl != "Neutral":
                            line += f" ({el['reason']})"
                        st.markdown(line)

                # Pull tendency warning
                if script.get("pull_warning"):
                    st.markdown(f"**{script['pull_warning']}**")


def page_scouting(data):
    st.header("Opponent Scouting")

    tm = _load_truemedia()
    # Get all TrueMedia team names
    all_tm_teams = set()
    for role in ["hitting", "pitching"]:
        for key, df in tm[role].items():
            if "newestTeamName" in df.columns:
                all_tm_teams.update(df["newestTeamName"].dropna().unique())
    all_tm_teams.discard("Davidson College")
    all_tm_teams = sorted(all_tm_teams)

    if not all_tm_teams:
        st.info("No TrueMedia data found.")
        return

    team = st.selectbox("Opponent", all_tm_teams, key="sc_team_tm")

    tab_overview, tab_hitters, tab_pitchers, tab_catchers, tab_gameplan = st.tabs([
        "Team Overview", "Their Hitters", "Their Pitchers", "Their Catchers", "Game Plan"
    ])

    # ── Tab 1: Team Overview ──
    with tab_overview:
        _scouting_team_overview(tm, team)

    # ── Tab 2: Their Hitters ──
    with tab_hitters:
        _scouting_hitter_report(tm, team, data)

    # ── Tab 3: Their Pitchers ──
    with tab_pitchers:
        _scouting_pitcher_report(tm, team, data)

    # ── Tab 4: Their Catchers ──
    with tab_catchers:
        _scouting_catcher_report(tm, team)

    # ── Tab 5: Game Plan ──
    with tab_gameplan:
        _game_plan_tab(tm, team, data)


def _scouting_team_overview(tm, team):
    """Team Overview tab — aggregate offense + pitching metrics with D1 rankings."""
    h_cnt = _tm_team(tm["hitting"]["counting"], team)
    h_rate = _tm_team(tm["hitting"]["rate"], team)
    p_trad = _tm_team(tm["pitching"]["traditional"], team)

    # Build team-level aggregates across ALL D1 teams for ranking context
    all_h_cnt = tm["hitting"]["counting"]
    all_p_trad = tm["pitching"]["traditional"]

    @st.cache_data(show_spinner=False)
    def _team_agg_ranks(_h_cnt_json, _p_trad_json):
        """Compute team-level aggregates for all teams and return rank context."""
        h_cnt_all = pd.read_json(_h_cnt_json) if _h_cnt_json else pd.DataFrame()
        p_trad_all = pd.read_json(_p_trad_json) if _p_trad_json else pd.DataFrame()

        team_off = {}
        if not h_cnt_all.empty:
            for t, grp in h_cnt_all.groupby("newestTeamName"):
                pa = grp["PA"].sum()
                ab = grp["AB"].sum()
                team_off[t] = {
                    "BA": grp["H"].sum() / max(ab, 1),
                    "HR": grp["HR"].sum(),
                    "SB": grp["SB"].sum(),
                    "K%": grp["K"].sum() / max(pa, 1) * 100,
                    "BB%": grp["BB"].sum() / max(pa, 1) * 100,
                }

        team_pit = {}
        if not p_trad_all.empty:
            for t, grp in p_trad_all.groupby("newestTeamName"):
                ip = grp["IP"].sum()
                team_pit[t] = {
                    "ERA": grp["ER"].sum() / max(ip, 1) * 9 if "ER" in grp.columns else np.nan,
                    "K/9": grp["K"].sum() / max(ip, 1) * 9,
                    "BB/9": grp["BB"].sum() / max(ip, 1) * 9,
                    "IP": ip,
                }

        return team_off, team_pit

    h_json = all_h_cnt.to_json() if not all_h_cnt.empty else ""
    p_json = all_p_trad.to_json() if not all_p_trad.empty else ""
    team_off_all, team_pit_all = _team_agg_ranks(h_json, p_json)
    n_teams = max(len(team_off_all), len(team_pit_all))

    # ── Team Narrative ──
    team_narrative = []
    if team in team_off_all:
        off = team_off_all[team]
        # Rank this team's offense
        ba_rank = sum(1 for t in team_off_all.values() if t["BA"] > off["BA"]) + 1
        hr_rank = sum(1 for t in team_off_all.values() if t["HR"] > off["HR"]) + 1
        k_rank = sum(1 for t in team_off_all.values() if t["K%"] < off["K%"]) + 1  # lower is better
        team_narrative.append(
            f"**Offense:** {team} hits **.{int(off['BA']*1000):03d}** as a team "
            f"(#{ba_rank} of {n_teams} D1 teams), with **{off['HR']}** HR (#{hr_rank}) "
            f"and a **{off['K%']:.1f}%** K rate (#{k_rank})."
        )
        if off["K%"] > 22:
            team_narrative.append("This lineup is strikeout-prone — expand the zone and use putaway pitches.")
        elif off["BB%"] > 10:
            team_narrative.append("A patient lineup — throw strikes early and avoid falling behind.")
        if off["HR"] > 40:
            team_narrative.append("Power-heavy — keep the ball down and limit mistakes over the plate.")
        if off["SB"] > 50:
            team_narrative.append(f"Speed threat ({off['SB']} SB) — quick deliveries and catcher readiness critical.")

    if team in team_pit_all:
        pit = team_pit_all[team]
        era_rank = sum(1 for t in team_pit_all.values() if t["ERA"] < pit["ERA"]) + 1  # lower is better
        k9_rank = sum(1 for t in team_pit_all.values() if t["K/9"] > pit["K/9"]) + 1
        team_narrative.append(
            f"**Pitching:** Staff ERA of **{pit['ERA']:.2f}** (#{era_rank} of {n_teams}), "
            f"**{pit['K/9']:.1f}** K/9 (#{k9_rank}), **{pit['BB/9']:.1f}** BB/9."
        )
        if pit["ERA"] > 5.0:
            team_narrative.append("Hittable staff — be aggressive and attack early.")
        elif pit["ERA"] < 3.5:
            team_narrative.append("Elite pitching staff — need disciplined, quality at-bats.")
        if pit["BB/9"] > 4.0:
            team_narrative.append("Control issues — work deep counts and take walks.")

    if team_narrative:
        st.markdown("\n\n".join(team_narrative))
    else:
        st.info("No aggregate data available for this team.")

    # ── Top Players Tables ──
    col_th, col_tp = st.columns(2)
    with col_th:
        section_header("Top Hitters (by PA)")
        if not h_rate.empty:
            all_rate = tm["hitting"]["rate"]
            d = h_rate[["playerFullName", "PA", "BA", "OBP", "SLG", "OPS"]].copy()
            d = d.sort_values("PA", ascending=False).head(10)
            # Add D1 OPS percentile
            d["OPS Pctile"] = d["OPS"].apply(
                lambda x: int(percentileofscore(all_rate["OPS"].dropna(), x, kind='rank')) if pd.notna(x) else np.nan
            )
            d.columns = ["Player", "PA", "BA", "OBP", "SLG", "OPS", "OPS %ile"]
            for c in ["BA", "OBP", "SLG", "OPS"]:
                d[c] = d[c].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
            d["OPS %ile"] = d["OPS %ile"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")
            st.dataframe(d, use_container_width=True, hide_index=True)
    with col_tp:
        section_header("Top Pitchers (by IP)")
        if not p_trad.empty:
            d = p_trad[["playerFullName", "GS", "IP", "ERA", "FIP", "WHIP", "K/9"]].copy()
            d = d.sort_values("IP", ascending=False).head(10)
            # Add D1 ERA percentile (inverted — lower ERA = higher percentile)
            all_trad = tm["pitching"]["traditional"]
            d["ERA Pctile"] = d["ERA"].apply(
                lambda x: int(100 - percentileofscore(all_trad["ERA"].dropna(), x, kind='rank')) if pd.notna(x) else np.nan
            )
            d.columns = ["Player", "GS", "IP", "ERA", "FIP", "WHIP", "K/9", "ERA %ile"]
            for c in ["ERA", "FIP", "WHIP"]:
                d[c] = d[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            d["K/9"] = d["K/9"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
            d["ERA %ile"] = d["ERA %ile"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")
            st.dataframe(d, use_container_width=True, hide_index=True)

    # ── Feature #9: Lineup Builder / Batting Order Analysis ──
    section_header("Lineup Analysis")
    h_exit = _tm_team(tm["hitting"]["exit"], team)
    h_speed = _tm_team(tm["hitting"]["speed"], team)
    h_prates = _tm_team(tm["hitting"]["pitch_rates"], team)
    h_htypes = _tm_team(tm["hitting"]["hit_types"], team)
    h_hlocs = _tm_team(tm["hitting"]["hit_locations"], team)

    if not h_rate.empty and not h_cnt.empty:
        # Build unified lineup dataframe
        lineup = h_cnt[["playerFullName", "PA", "AB", "H", "HR", "RBI", "BB", "K", "SB"]].copy()
        lineup = lineup[lineup["PA"] >= 10].copy()  # min 10 PA

        # Merge rate stats
        if not h_rate.empty:
            rate_cols = ["playerFullName"]
            for c in ["BA", "OBP", "SLG", "OPS", "ISO", "WOBA", "BABIP"]:
                if c in h_rate.columns:
                    rate_cols.append(c)
            lineup = lineup.merge(h_rate[rate_cols], on="playerFullName", how="left")
            if "WOBA" in lineup.columns:
                lineup = lineup.rename(columns={"WOBA": "wOBA"})

        # Merge exit data
        if not h_exit.empty and "ExitVel" in h_exit.columns:
            ev_cols = ["playerFullName"]
            for c in ["ExitVel", "Barrel%", "HardHit%"]:
                if c in h_exit.columns:
                    ev_cols.append(c)
            lineup = lineup.merge(h_exit[ev_cols], on="playerFullName", how="left")

        # Merge speed
        if not h_speed.empty and "SpeedScore" in h_speed.columns:
            lineup = lineup.merge(h_speed[["playerFullName", "SpeedScore"]], on="playerFullName", how="left")

        # Merge K%, BB% from pitch rates
        if not h_prates.empty:
            pr_cols = ["playerFullName"]
            for c in ["K%", "BB%", "Chase%", "SwStrk%"]:
                if c in h_prates.columns:
                    pr_cols.append(c)
            lineup = lineup.merge(h_prates[pr_cols], on="playerFullName", how="left")

        # Compute a composite "production score" for sorting
        lineup["ProdScore"] = 0.0
        if "wOBA" in lineup.columns:
            woba_med = lineup["wOBA"].median()
            lineup["ProdScore"] += lineup["wOBA"].fillna(woba_med) * 100
        elif "OPS" in lineup.columns:
            ops_med = lineup["OPS"].median()
            lineup["ProdScore"] += lineup["OPS"].fillna(ops_med) * 50
        if "ExitVel" in lineup.columns:
            lineup["ProdScore"] += lineup["ExitVel"].fillna(80) * 0.3
        if "SpeedScore" in lineup.columns:
            lineup["ProdScore"] += lineup["SpeedScore"].fillna(3) * 2

        lineup = lineup.sort_values("ProdScore", ascending=False)

        # Classify hitters into roles
        def _classify_hitter(row):
            roles = []
            if pd.notna(row.get("ISO")) and row["ISO"] >= 0.180:
                roles.append("Power")
            elif pd.notna(row.get("HR")) and row["HR"] >= 5:
                roles.append("Power")
            if pd.notna(row.get("SpeedScore")) and row["SpeedScore"] >= 5.0:
                roles.append("Speed")
            elif pd.notna(row.get("SB")) and row["SB"] >= 5:
                roles.append("Speed")
            if pd.notna(row.get("BB%")) and row["BB%"] >= 10:
                roles.append("Patient")
            if pd.notna(row.get("Chase%")) and row["Chase%"] <= 20:
                roles.append("Disciplined")
            if pd.notna(row.get("BA")) and row["BA"] >= 0.300:
                roles.append("Contact")
            elif pd.notna(row.get("K%")) and row["K%"] <= 15:
                roles.append("Contact")
            if pd.notna(row.get("ExitVel")) and row["ExitVel"] >= 90:
                roles.append("Hard-Hit")
            if not roles:
                roles.append("Utility")
            return ", ".join(roles)

        lineup["Role"] = lineup.apply(_classify_hitter, axis=1)

        # Display lineup table
        display_cols = ["playerFullName", "PA"]
        disp_names = {"playerFullName": "Player", "PA": "PA"}
        for c, name, fmt in [
            ("BA", "BA", ".3f"), ("OBP", "OBP", ".3f"), ("SLG", "SLG", ".3f"),
            ("wOBA", "wOBA", ".3f"), ("ISO", "ISO", ".3f"),
            ("HR", "HR", "d"), ("SB", "SB", "d"),
            ("ExitVel", "EV", ".1f"), ("Barrel%", "Brl%", ".1f"),
            ("SpeedScore", "Spd", ".1f"), ("Chase%", "Chase%", ".1f"),
        ]:
            if c in lineup.columns:
                display_cols.append(c)
                disp_names[c] = name
        display_cols.append("Role")
        disp_names["Role"] = "Profile"

        disp = lineup[display_cols].head(15).copy()
        disp = disp.rename(columns=disp_names)
        # Format numeric columns
        for c in disp.columns:
            if c in ["BA", "OBP", "SLG", "wOBA", "ISO"]:
                disp[c] = disp[c].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
            elif c in ["EV", "Spd", "Brl%", "Chase%"]:
                disp[c] = disp[c].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
            elif c in ["HR", "SB", "PA"]:
                disp[c] = disp[c].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")
        st.dataframe(disp, use_container_width=True, hide_index=True)

        # Lineup insights
        insights = []
        power_count = lineup[lineup.get("Role", pd.Series(dtype=str)).str.contains("Power", na=False)].shape[0] if "Role" in lineup.columns else 0
        speed_count = lineup[lineup.get("Role", pd.Series(dtype=str)).str.contains("Speed", na=False)].shape[0] if "Role" in lineup.columns else 0
        patient_count = lineup[lineup.get("Role", pd.Series(dtype=str)).str.contains("Patient|Disciplined", na=False)].shape[0] if "Role" in lineup.columns else 0

        if power_count >= 4:
            insights.append(f"⚡ **Power-heavy lineup** ({power_count} power hitters) — keep the ball down, avoid center-cut fastballs.")
        elif power_count <= 1:
            insights.append("Lacks power — pitchers can pitch to contact and let defense work.")
        if speed_count >= 3:
            insights.append(f"🏃 **Speed threat** ({speed_count} speed players) — quick deliveries, catcher awareness critical.")
        if patient_count >= 4:
            insights.append(f"👁️ **Disciplined lineup** ({patient_count} patient hitters) — must throw strikes early, avoid falling behind.")
        elif patient_count <= 1:
            insights.append("Aggressive lineup — expand the zone with offspeed, use putaway pitches.")

        if "Chase%" in lineup.columns:
            avg_chase = lineup["Chase%"].mean()
            if pd.notna(avg_chase):
                if avg_chase > 28:
                    insights.append(f"High team chase rate ({avg_chase:.1f}%) — work off the plate aggressively.")
                elif avg_chase < 20:
                    insights.append(f"Low team chase rate ({avg_chase:.1f}%) — must compete in the zone.")

        for ins in insights:
            st.markdown(ins)
    else:
        st.info("Insufficient hitting data for lineup analysis.")

    # ── Feature #10: Pitching Staff Depth Chart ──
    section_header("Pitching Staff Depth Chart")
    if not p_trad.empty:
        p_rate = _tm_team(tm["pitching"]["rate"], team)
        p_mov = _tm_team(tm["pitching"]["movement"], team)
        p_pit_rates = _tm_team(tm["pitching"]["pitch_rates"], team)
        p_exit_data = _tm_team(tm["pitching"]["exit"], team)
        p_bids_team = _tm_team(tm["pitching"]["bids"], team)

        staff = p_trad[["playerFullName", "G", "GS", "IP", "ERA", "FIP", "WHIP", "K/9", "BB/9"]].copy()
        staff = staff[staff["IP"] >= 1].copy()

        # Merge velocity from movement
        if not p_mov.empty and "Vel" in p_mov.columns:
            vel_cols = ["playerFullName"]
            for c in ["Vel", "MxVel"]:
                if c in p_mov.columns:
                    vel_cols.append(c)
            staff = staff.merge(p_mov[vel_cols], on="playerFullName", how="left")

        # Merge SwStrk% from pitch rates
        if not p_pit_rates.empty and "SwStrk%" in p_pit_rates.columns:
            staff = staff.merge(p_pit_rates[["playerFullName", "SwStrk%"]], on="playerFullName", how="left")

        # Merge ExitVel against
        if not p_exit_data.empty and "ExitVel" in p_exit_data.columns:
            staff = staff.merge(
                p_exit_data[["playerFullName", "ExitVel"]].rename(columns={"ExitVel": "EV Against"}),
                on="playerFullName", how="left"
            )

        # Classify starters vs relievers
        def _classify_pitcher(row):
            gs = row.get("GS", 0) or 0
            g = row.get("G", 0) or 0
            ip = row.get("IP", 0) or 0
            if gs >= 3 or (gs > 0 and gs / max(g, 1) > 0.5):
                return "Starter"
            elif ip >= 5 or g >= 5:
                return "Reliever"
            else:
                return "Spot/Mop-up"

        staff["Role"] = staff.apply(_classify_pitcher, axis=1)

        # Separate and display starters
        starters = staff[staff["Role"] == "Starter"].sort_values("IP", ascending=False)
        relievers = staff[staff["Role"].isin(["Reliever", "Spot/Mop-up"])].sort_values("IP", ascending=False)

        col_s, col_r = st.columns(2)

        def _format_staff_table(df, label):
            display = df.copy()
            cols_show = ["playerFullName", "G", "GS", "IP", "ERA", "FIP", "WHIP", "K/9", "BB/9"]
            col_names = {"playerFullName": "Player", "G": "G", "GS": "GS", "IP": "IP",
                         "ERA": "ERA", "FIP": "FIP", "WHIP": "WHIP", "K/9": "K/9", "BB/9": "BB/9"}
            if "Vel" in display.columns:
                cols_show.append("Vel")
                col_names["Vel"] = "Velo"
            if "SwStrk%" in display.columns:
                cols_show.append("SwStrk%")
                col_names["SwStrk%"] = "SwStr%"
            if "EV Against" in display.columns:
                cols_show.append("EV Against")
                col_names["EV Against"] = "EV Agn"

            avail = [c for c in cols_show if c in display.columns]
            display = display[avail].rename(columns=col_names)
            for c in ["ERA", "FIP", "WHIP"]:
                if c in display.columns:
                    display[c] = display[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            for c in ["K/9", "BB/9", "Velo", "SwStr%", "EV Agn"]:
                if c in display.columns:
                    display[c] = display[c].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
            for c in ["G", "GS"]:
                if c in display.columns:
                    display[c] = display[c].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")
            if "IP" in display.columns:
                display["IP"] = display["IP"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
            return display

        with col_s:
            st.markdown(f"**Starters ({len(starters)})**")
            if not starters.empty:
                st.dataframe(_format_staff_table(starters, "Starters"), use_container_width=True, hide_index=True)
            else:
                st.caption("No identified starters.")

        with col_r:
            st.markdown(f"**Bullpen ({len(relievers)})**")
            if not relievers.empty:
                st.dataframe(_format_staff_table(relievers, "Relievers"), use_container_width=True, hide_index=True)
            else:
                st.caption("No identified relievers.")

        # Staff-level insights
        staff_insights = []
        if not starters.empty:
            avg_era = starters["ERA"].mean()
            avg_vel = starters["Vel"].mean() if "Vel" in starters.columns else np.nan
            if pd.notna(avg_era):
                staff_insights.append(f"Rotation ERA: **{avg_era:.2f}**")
            if pd.notna(avg_vel):
                staff_insights.append(f"Rotation avg velo: **{avg_vel:.1f}** mph")
            top_sp = starters.iloc[0]
            staff_insights.append(
                f"Ace: **{top_sp['playerFullName']}** ({top_sp['IP']:.1f} IP, {top_sp['ERA']:.2f} ERA)"
            )
        if not relievers.empty:
            avg_rel_era = relievers["ERA"].mean()
            if pd.notna(avg_rel_era):
                staff_insights.append(f"Bullpen ERA: **{avg_rel_era:.2f}**")
            if "SwStrk%" in relievers.columns:
                best_sw = relievers.loc[relievers["SwStrk%"].idxmax()] if relievers["SwStrk%"].notna().any() else None
                if best_sw is not None and pd.notna(best_sw.get("SwStrk%")):
                    staff_insights.append(
                        f"Best swing & miss: **{best_sw['playerFullName']}** ({best_sw['SwStrk%']:.1f}% SwStr)")

        if staff_insights:
            st.markdown(" | ".join(staff_insights))
    else:
        st.info("No pitching data available for staff depth chart.")

    # ── Feature #11: Team Tendencies Dashboard ──
    section_header("Team Tendencies")

    # Load additional team-level data
    h_htypes_team = _tm_team(tm["hitting"]["hit_types"], team)
    h_hlocs_team = _tm_team(tm["hitting"]["hit_locations"], team)
    h_prates_team = _tm_team(tm["hitting"]["pitch_rates"], team)
    h_exit_team = _tm_team(tm["hitting"]["exit"], team)
    h_sb_team = _tm_team(tm["hitting"]["stolen_bases"], team)
    h_speed_team = _tm_team(tm["hitting"]["speed"], team)
    p_htypes_team = _tm_team(tm["pitching"]["hit_types"], team)
    p_prates_team = _tm_team(tm["pitching"]["pitch_rates"], team)
    p_ptypes_team = _tm_team(tm["pitching"]["pitch_types"], team)

    tend_col1, tend_col2 = st.columns(2)

    # ── Offensive Tendencies ──
    with tend_col1:
        st.markdown("**Offensive Tendencies**")

        # Batted ball profile (team averages)
        if not h_htypes_team.empty:
            bb_data = []
            for lbl, col in [("GB%", "Ground%"), ("FB%", "Fly%"), ("LD%", "Line%"), ("Popup%", "Popup%")]:
                if col in h_htypes_team.columns:
                    avg = h_htypes_team[col].mean()
                    if pd.notna(avg):
                        bb_data.append({"Type": lbl, "Team Avg": f"{avg:.1f}%"})
            if bb_data:
                st.dataframe(pd.DataFrame(bb_data), use_container_width=True, hide_index=True)

        # Spray direction (team averages)
        if not h_hlocs_team.empty:
            spray_data = []
            for lbl, col in [("Pull%", "HPull%"), ("Center%", "HCtr%"), ("Oppo%", "HOppFld%")]:
                if col in h_hlocs_team.columns:
                    avg = h_hlocs_team[col].mean()
                    if pd.notna(avg):
                        spray_data.append({"Direction": lbl, "Team Avg": f"{avg:.1f}%"})
            if spray_data:
                st.dataframe(pd.DataFrame(spray_data), use_container_width=True, hide_index=True)

        # Plate discipline (team averages)
        if not h_prates_team.empty:
            disc_data = []
            for lbl, col in [("Swing%", "Swing%"), ("Contact%", "Contact%"), ("Chase%", "Chase%"),
                              ("SwStrk%", "SwStrk%"), ("InZone Swing%", "Z-Swing%")]:
                if col in h_prates_team.columns:
                    avg = h_prates_team[col].mean()
                    if pd.notna(avg):
                        disc_data.append({"Metric": lbl, "Team Avg": f"{avg:.1f}%"})
            if disc_data:
                st.dataframe(pd.DataFrame(disc_data), use_container_width=True, hide_index=True)

        # Exit velocity / quality of contact
        if not h_exit_team.empty:
            qc_data = []
            for lbl, col in [("Avg Exit Velo", "ExitVel"), ("Barrel%", "Barrel%"),
                              ("Hard Hit%", "HardHit%"), ("Sweet Spot%", "SweetSpot%")]:
                if col in h_exit_team.columns:
                    avg = h_exit_team[col].mean()
                    if pd.notna(avg):
                        fmt = f"{avg:.1f}" if "%" not in lbl else f"{avg:.1f}%"
                        if lbl == "Avg Exit Velo":
                            fmt = f"{avg:.1f} mph"
                        qc_data.append({"Metric": lbl, "Team Avg": fmt})
            if qc_data:
                st.dataframe(pd.DataFrame(qc_data), use_container_width=True, hide_index=True)

        # Baserunning tendencies
        sb_insights = []
        if not h_sb_team.empty:
            for col in ["SB2%", "SB3%"]:
                if col in h_sb_team.columns:
                    avg = h_sb_team[col].mean()
                    if pd.notna(avg):
                        sb_insights.append(f"{col}: {avg:.1f}%")
        if not h_speed_team.empty and "SpeedScore" in h_speed_team.columns:
            avg_spd = h_speed_team["SpeedScore"].mean()
            if pd.notna(avg_spd):
                sb_insights.append(f"Avg Speed Score: {avg_spd:.1f}")
        if sb_insights:
            st.caption("🏃 Baserunning: " + " | ".join(sb_insights))

    # ── Pitching Staff Tendencies ──
    with tend_col2:
        st.markdown("**Pitching Staff Tendencies**")

        # Pitch mix (team average)
        if not p_ptypes_team.empty:
            pitch_mix = []
            for lbl, col in [("Fastball", "Fastball%"), ("Sinker", "Sinker%"), ("Cutter", "Cutter%"),
                              ("Slider", "Slider%"), ("Curveball", "Curveball%"), ("Changeup", "Changeup%"),
                              ("Sweeper", "Sweeper%"), ("Splitter", "Splitter%")]:
                if col in p_ptypes_team.columns:
                    avg = p_ptypes_team[col].mean()
                    if pd.notna(avg) and avg > 0.5:
                        pitch_mix.append({"Pitch": lbl, "Usage%": f"{avg:.1f}%"})
            if pitch_mix:
                pitch_mix_df = pd.DataFrame(pitch_mix).sort_values("Usage%", ascending=False, key=lambda s: s.str.rstrip("%").astype(float))
                st.dataframe(pitch_mix_df, use_container_width=True, hide_index=True)

        # Batted ball types allowed (pitching staff)
        if not p_htypes_team.empty:
            pbb_data = []
            for lbl, col in [("GB%", "Ground%"), ("FB%", "Fly%"), ("LD%", "Line%"), ("Popup%", "Popup%")]:
                if col in p_htypes_team.columns:
                    avg = p_htypes_team[col].mean()
                    if pd.notna(avg):
                        pbb_data.append({"Type": lbl, "Staff Avg": f"{avg:.1f}%"})
            if pbb_data:
                st.dataframe(pd.DataFrame(pbb_data), use_container_width=True, hide_index=True)

        # Staff pitch discipline induced
        if not p_prates_team.empty:
            pdisc_data = []
            for lbl, col in [("Miss%", "Miss%"), ("Chase%", "Chase%"), ("SwStrk%", "SwStrk%"),
                              ("CompLoc%", "CompLoc%"), ("InZone%", "InZone%")]:
                if col in p_prates_team.columns:
                    avg = p_prates_team[col].mean()
                    if pd.notna(avg):
                        pdisc_data.append({"Metric": lbl, "Staff Avg": f"{avg:.1f}%"})
            if pdisc_data:
                st.dataframe(pd.DataFrame(pdisc_data), use_container_width=True, hide_index=True)

    # Team tendency narrative
    tend_narrative = []
    if not h_htypes_team.empty and "Ground%" in h_htypes_team.columns:
        avg_gb = h_htypes_team["Ground%"].mean()
        avg_fb = h_htypes_team["Fly%"].mean() if "Fly%" in h_htypes_team.columns else np.nan
        if pd.notna(avg_gb) and avg_gb > 48:
            tend_narrative.append(f"**Ground-ball hitting team** ({avg_gb:.1f}% GB) — infield defense matters, look for double-play opportunities.")
        elif pd.notna(avg_fb) and avg_fb > 38:
            tend_narrative.append(f"**Fly-ball hitting team** ({avg_fb:.1f}% FB) — outfield positioning critical, HR risk in elevated pitches.")

    if not h_hlocs_team.empty and "HPull%" in h_hlocs_team.columns:
        avg_pull = h_hlocs_team["HPull%"].mean()
        avg_oppo = h_hlocs_team["HOppFld%"].mean() if "HOppFld%" in h_hlocs_team.columns else np.nan
        if pd.notna(avg_pull) and avg_pull > 45:
            tend_narrative.append(f"**Pull-heavy offense** ({avg_pull:.1f}% Pull) — shift infield, work away.")
        elif pd.notna(avg_oppo) and avg_oppo > 28:
            tend_narrative.append(f"**Good opposite-field approach** ({avg_oppo:.1f}% Oppo) — this team uses the whole field.")

    if not h_prates_team.empty and "Chase%" in h_prates_team.columns:
        avg_chase = h_prates_team["Chase%"].mean()
        avg_contact = h_prates_team["Contact%"].mean() if "Contact%" in h_prates_team.columns else np.nan
        if pd.notna(avg_chase) and avg_chase > 28:
            tend_narrative.append(f"**Chase-prone lineup** ({avg_chase:.1f}% Chase) — expand zone aggressively with breaking balls.")
        elif pd.notna(avg_chase) and avg_chase < 20:
            tend_narrative.append(f"**Extremely selective lineup** ({avg_chase:.1f}% Chase) — compete in the zone, don't waste pitches.")
        if pd.notna(avg_contact) and avg_contact > 82:
            tend_narrative.append(f"**High-contact team** ({avg_contact:.1f}% Contact) — need swing-and-miss stuff or defensive excellence.")

    if not p_prates_team.empty and "SwStrk%" in p_prates_team.columns:
        avg_swstrk = p_prates_team["SwStrk%"].mean()
        if pd.notna(avg_swstrk):
            if avg_swstrk > 12:
                tend_narrative.append(f"**High-strikeout staff** ({avg_swstrk:.1f}% SwStr) — two-strike approach critical, protect the plate.")
            elif avg_swstrk < 8:
                tend_narrative.append(f"**Low swing-and-miss staff** ({avg_swstrk:.1f}% SwStr) — hittable, be aggressive early in counts.")

    if tend_narrative:
        st.markdown("---")
        st.markdown("**Key Tendencies:**")
        for t in tend_narrative:
            st.markdown(f"- {t}")


def _scouting_hitter_report(tm, team, trackman_data):
    """Their Hitters tab — individual deep-dive scouting report with percentile context."""
    h_rate = _tm_team(tm["hitting"]["rate"], team)
    if h_rate.empty:
        st.info("No hitting data for this team.")
        return
    hitters = sorted(h_rate["playerFullName"].unique())
    hitter = st.selectbox("Select Hitter", hitters, key="sc_hitter")

    # Get this player's data from all tables
    rate = _tm_player(h_rate, hitter)
    cnt = _tm_player(_tm_team(tm["hitting"]["counting"], team), hitter)
    exit_d = _tm_player(_tm_team(tm["hitting"]["exit"], team), hitter)
    xrate = _tm_player(_tm_team(tm["hitting"]["expected_rate"], team), hitter)
    ht = _tm_player(_tm_team(tm["hitting"]["hit_types"], team), hitter)
    hl = _tm_player(_tm_team(tm["hitting"]["hit_locations"], team), hitter)
    pr = _tm_player(_tm_team(tm["hitting"]["pitch_rates"], team), hitter)
    pt = _tm_player(_tm_team(tm["hitting"]["pitch_types"], team), hitter)
    pl = _tm_player(_tm_team(tm["hitting"]["pitch_locations"], team), hitter)
    spd = _tm_player(_tm_team(tm["hitting"]["speed"], team), hitter)
    sb = _tm_player(_tm_team(tm["hitting"]["stolen_bases"], team), hitter)
    hrs = _tm_player(_tm_team(tm["hitting"]["home_runs"], team), hitter)
    h_pcounts = _tm_player(_tm_team(tm["hitting"]["pitch_counts"], team), hitter)
    h_ptcounts = _tm_player(_tm_team(tm["hitting"]["pitch_type_counts"], team), hitter)
    h_re = _tm_player(_tm_team(tm["hitting"]["run_expectancy"], team), hitter)
    h_swpct = _tm_player(_tm_team(tm["hitting"]["swing_pct"], team), hitter)
    h_swstats = _tm_player(_tm_team(tm["hitting"]["swing_stats"], team), hitter)

    # All D1 data for percentile context
    all_h_rate = tm["hitting"]["rate"]
    all_h_exit = tm["hitting"]["exit"]
    all_h_pr = tm["hitting"]["pitch_rates"]
    all_h_ht = tm["hitting"]["hit_types"]
    all_h_hl = tm["hitting"]["hit_locations"]
    all_h_spd = tm["hitting"]["speed"]
    all_h_re = tm["hitting"]["run_expectancy"]
    all_h_swpct = tm["hitting"]["swing_pct"]
    all_h_swstats = tm["hitting"]["swing_stats"]

    # Header
    pos = rate.iloc[0].get("pos", "?") if not rate.empty else "?"
    bats = rate.iloc[0].get("batsHand", "?") if not rate.empty else "?"
    g = _safe_val(cnt, "G", "d")
    pa = _safe_val(cnt, "PA", "d")
    st.markdown(f"### {hitter}")
    st.caption(f"{pos} | Bats: {bats} | G: {g} | PA: {pa}")

    # ── Player Narrative ──
    narrative = _hitter_narrative(hitter, rate, exit_d, pr, ht, hl, spd,
                                  all_h_rate, all_h_exit, all_h_pr)
    st.markdown(narrative)

    n_hitters = len(all_h_rate)

    # ── Percentile Rankings (Savant-style bars) ──
    section_header("Percentile Rankings")
    st.caption(f"vs. {n_hitters:,} D1 hitters")
    hitting_metrics = [
        ("OPS", _safe_num(rate, "OPS"), _tm_pctile(rate, "OPS", all_h_rate), ".3f", True),
        ("wOBA", _safe_num(rate, "WOBA"), _tm_pctile(rate, "WOBA", all_h_rate), ".3f", True),
        ("BA", _safe_num(rate, "BA"), _tm_pctile(rate, "BA", all_h_rate), ".3f", True),
        ("SLG", _safe_num(rate, "SLG"), _tm_pctile(rate, "SLG", all_h_rate), ".3f", True),
        ("ISO", _safe_num(rate, "ISO"), _tm_pctile(rate, "ISO", all_h_rate), ".3f", True),
        ("Exit Velo", _safe_num(exit_d, "ExitVel"), _tm_pctile(exit_d, "ExitVel", all_h_exit), ".1f", True),
        ("Barrel %", _safe_num(exit_d, "Barrel%"), _tm_pctile(exit_d, "Barrel%", all_h_exit), ".1f", True),
        ("Hard Hit %", _safe_num(exit_d, "Hit95+%"), _tm_pctile(exit_d, "Hit95+%", all_h_exit), ".1f", True),
        ("K %", _safe_num(rate, "K%"), _tm_pctile(rate, "K%", all_h_rate), ".1f", False),
        ("BB %", _safe_num(rate, "BB%"), _tm_pctile(rate, "BB%", all_h_rate), ".1f", True),
        ("Chase %", _safe_num(pr, "Chase%"), _tm_pctile(pr, "Chase%", all_h_pr), ".1f", False),
        ("Whiff %", _safe_num(pr, "SwStrk%"), _tm_pctile(pr, "SwStrk%", all_h_pr), ".1f", False),
        ("Contact %", _safe_num(pr, "Contact%"), _tm_pctile(pr, "Contact%", all_h_pr), ".1f", True),
        ("Speed Score", _safe_num(spd, "SpeedScore"), _tm_pctile(spd, "SpeedScore", all_h_spd), ".1f", True),
    ]
    # Filter out metrics with nan values
    hitting_metrics = [(l, v, p, f, h) for l, v, p, f, h in hitting_metrics if not pd.isna(v)]
    render_savant_percentile_section(hitting_metrics)

    # ── Batted Ball & Spray Profile ──
    col_bb, col_sp = st.columns(2)
    with col_bb:
        if not ht.empty:
            section_header("Batted Ball Types")
            bb_metrics = [
                ("GB %", _safe_num(ht, "Ground%"), _tm_pctile(ht, "Ground%", all_h_ht), ".1f", True),
                ("FB %", _safe_num(ht, "Fly%"), _tm_pctile(ht, "Fly%", all_h_ht), ".1f", True),
                ("LD %", _safe_num(ht, "Line%"), _tm_pctile(ht, "Line%", all_h_ht), ".1f", True),
                ("Popup %", _safe_num(ht, "Popup%"), _tm_pctile(ht, "Popup%", all_h_ht), ".1f", True),
            ]
            bb_metrics = [(l, v, p, f, h) for l, v, p, f, h in bb_metrics if not pd.isna(v)]
            render_savant_percentile_section(bb_metrics, legend=("LESS OFTEN", "AVERAGE", "MORE OFTEN"))
    with col_sp:
        # Feature #7: Spray Chart Visualization (5-zone field diagram)
        if not hl.empty:
            section_header("Spray Chart")
            farlft = _safe_num(hl, "HFarLft%")
            lftctr = _safe_num(hl, "HLftCtr%")
            deadctr = _safe_num(hl, "HDeadCtr%")
            rtctr = _safe_num(hl, "HRtCtr%")
            farrt = _safe_num(hl, "HFarRt%")
            zones = [farlft, lftctr, deadctr, rtctr, farrt]
            zone_labels = ["Far Left", "Left-Ctr", "Dead Ctr", "Right-Ctr", "Far Right"]
            if any(not pd.isna(z) for z in zones):
                # Build a polar/fan spray chart (viewed from above, catcher's perspective)
                # Angles: 0° = straight up (center field), negative = left, positive = right
                # Zone order: FarLft, LftCtr, DeadCtr, RtCtr, FarRt
                # From catcher view: FarLft = right side of chart, FarRt = left side
                angles_mid = [60, 30, 0, -30, -60]  # degrees from center field
                angle_width = 28
                fig_spray = go.Figure()

                # Draw foul lines and outfield arc for context
                fl_r = 115
                # Left field foul line (from home plate going up-right)
                fl_angle_l = math.radians(45)
                fig_spray.add_trace(go.Scatter(
                    x=[0, fl_r * math.sin(fl_angle_l)], y=[0, fl_r * math.cos(fl_angle_l)],
                    mode="lines", line=dict(color="rgba(0,0,0,0.15)", width=1, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                ))
                # Right field foul line
                fl_angle_r = math.radians(-45)
                fig_spray.add_trace(go.Scatter(
                    x=[0, fl_r * math.sin(fl_angle_r)], y=[0, fl_r * math.cos(fl_angle_r)],
                    mode="lines", line=dict(color="rgba(0,0,0,0.15)", width=1, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                ))
                # Outfield arc
                arc_pts = 30
                arc_x, arc_y = [], []
                for j in range(arc_pts + 1):
                    t = math.radians(-45 + 90 * j / arc_pts)
                    arc_x.append(fl_r * math.sin(t))
                    arc_y.append(fl_r * math.cos(t))
                fig_spray.add_trace(go.Scatter(
                    x=arc_x, y=arc_y, mode="lines",
                    line=dict(color="rgba(0,0,0,0.1)", width=1, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                ))

                max_val = max((z for z in zones if not pd.isna(z)), default=1)
                for i, (z, lbl, ang) in enumerate(zip(zones, zone_labels, angles_mid)):
                    if pd.isna(z):
                        continue
                    # Radius proportional to percentage (scale to fill chart)
                    r = z / max(max_val, 1) * 105
                    theta0 = math.radians(ang - angle_width / 2)
                    theta1 = math.radians(ang + angle_width / 2)
                    # Build wedge: sin for x, cos for y (baseball coordinates)
                    n_pts = 20
                    path_x = [0]
                    path_y = [0]
                    for j in range(n_pts + 1):
                        t = theta0 + (theta1 - theta0) * j / n_pts
                        path_x.append(r * math.sin(t))
                        path_y.append(r * math.cos(t))
                    path_x.append(0)
                    path_y.append(0)
                    # Color: darker = more hits
                    intensity = min(z / max(max_val, 1), 1.0)
                    r_c = int(220 - intensity * 180)
                    g_c = int(225 - intensity * 185)
                    b_c = int(240 - intensity * 50)
                    color = f"rgba({r_c},{g_c},{b_c},0.85)"
                    fig_spray.add_trace(go.Scatter(
                        x=path_x, y=path_y, fill="toself", fillcolor=color,
                        line=dict(color="white", width=2),
                        name=lbl, text=f"{lbl}: {z:.1f}%",
                        hoverinfo="text", showlegend=False,
                    ))
                    # Label
                    lbl_r = r * 0.55
                    lbl_t = math.radians(ang)
                    fig_spray.add_annotation(
                        x=lbl_r * math.sin(lbl_t), y=lbl_r * math.cos(lbl_t),
                        text=f"<b>{z:.0f}%</b>", showarrow=False,
                        font=dict(size=12, color="#1a1a2e"), bgcolor="rgba(255,255,255,0.8)",
                    )

                # Field labels
                fig_spray.add_annotation(x=-85, y=85, text="<b>LF</b>", showarrow=False,
                                         font=dict(size=10, color="rgba(0,0,0,0.3)"))
                fig_spray.add_annotation(x=0, y=118, text="<b>CF</b>", showarrow=False,
                                         font=dict(size=10, color="rgba(0,0,0,0.3)"))
                fig_spray.add_annotation(x=85, y=85, text="<b>RF</b>", showarrow=False,
                                         font=dict(size=10, color="rgba(0,0,0,0.3)"))

                fig_spray.update_layout(
                    **CHART_LAYOUT, height=360,
                    xaxis=dict(visible=False, range=[-130, 130]),
                    yaxis=dict(visible=False, range=[-10, 130], scaleanchor="x"),
                    showlegend=False,
                )
                st.plotly_chart(fig_spray, use_container_width=True)
                # Pull/Center/Oppo summary
                pull = _safe_num(hl, "HPull%")
                ctr = _safe_num(hl, "HCtr%")
                oppo = _safe_num(hl, "HOppFld%")
                parts = []
                if not pd.isna(pull):
                    parts.append(f"Pull: {pull:.1f}%")
                if not pd.isna(ctr):
                    parts.append(f"Center: {ctr:.1f}%")
                if not pd.isna(oppo):
                    parts.append(f"Oppo: {oppo:.1f}%")
                if parts:
                    st.caption(" | ".join(parts))

    # ── Feature #6: Pitch-Type Matchup Matrix ──
    if not pt.empty:
        section_header("Pitch Type Matchup")
        st.caption("What pitches this hitter sees and how often — plan your attack around weaknesses")
        pitch_cols = ["4Seam%", "Sink2Seam%", "Cutter%", "Slider%", "Curve%", "Change%", "Split%", "Sweeper%"]
        pitch_labels = ["4-Seam", "Sinker", "Cutter", "Slider", "Curve", "Change", "Splitter", "Sweeper"]
        count_cols = ["4Seam#", "Sink2Seam#", "Cutter#", "Slider#", "Curve#", "Change#", "Split#", "Sweeper#"]
        vals = []
        for pct_col, cnt_col, lbl in zip(pitch_cols, count_cols, pitch_labels):
            pct_v = pt.iloc[0].get(pct_col) if not pt.empty else None
            cnt_v = h_ptcounts.iloc[0].get(cnt_col) if not h_ptcounts.empty and cnt_col in h_ptcounts.columns else None
            if pct_v is not None and not pd.isna(pct_v) and pct_v > 0:
                row = {"Pitch": lbl, "% Seen": f"{pct_v:.1f}%"}
                if cnt_v is not None and not pd.isna(cnt_v):
                    row["Count"] = f"{int(cnt_v)}"
                else:
                    row["Count"] = "-"
                vals.append(row)
        if vals:
            # Bar chart + table side by side
            col_chart, col_tbl = st.columns([3, 2])
            with col_chart:
                vdf = pd.DataFrame(vals)
                pct_vals = [float(v["% Seen"].rstrip("%")) for v in vals]
                fig = go.Figure(go.Bar(
                    x=[v["Pitch"] for v in vals], y=pct_vals,
                    marker_color=[PITCH_COLORS.get(v["Pitch"].replace("-", ""), "#888") for v in vals],
                    text=[v["% Seen"] for v in vals], textposition="outside",
                ))
                fig.update_layout(**CHART_LAYOUT, height=280, yaxis_title="% Seen", showlegend=False,
                                  yaxis=dict(range=[0, max(pct_vals) * 1.3]))
                st.plotly_chart(fig, use_container_width=True)
            with col_tbl:
                st.dataframe(pd.DataFrame(vals), use_container_width=True, hide_index=True)

    # ── Power Profile ──
    if not hrs.empty:
        hr_val = _safe_num(hrs, "HR")
        if not pd.isna(hr_val) and hr_val > 0:
            section_header("Power Profile")
            all_h_hrs = tm["hitting"]["home_runs"]
            hr_metrics = [
                ("HR", _safe_num(hrs, "HR"), _tm_pctile(hrs, "HR", all_h_hrs), ".0f", True),
                ("HR/FB", _safe_num(hrs, "HR/FB"), _tm_pctile(hrs, "HR/FB", all_h_hrs), ".1f", True),
                ("HR Dist", _safe_num(hrs, "HRDst"), _tm_pctile(hrs, "HRDst", all_h_hrs), ".0f", True),
                ("FB Dist", _safe_num(hrs, "FBDst"), _tm_pctile(hrs, "FBDst", all_h_hrs), ".0f", True),
            ]
            hr_metrics = [(l, v, p, f, h) for l, v, p, f, h in hr_metrics if not pd.isna(v)]
            if hr_metrics:
                render_savant_percentile_section(hr_metrics)
            # HR direction
            hr_pull = _safe_num(hrs, "HRPull")
            hr_ctr = _safe_num(hrs, "HRCtr")
            hr_opp = _safe_num(hrs, "HROpp")
            if not pd.isna(hr_pull):
                st.caption(f"HR Direction: Pull {int(hr_pull)} | Center {int(hr_ctr) if not pd.isna(hr_ctr) else 0} | Oppo {int(hr_opp) if not pd.isna(hr_opp) else 0}")

    # ── Stolen Base Detail ──
    if not sb.empty:
        sb2 = _safe_num(sb, "SB2")
        sb3 = _safe_num(sb, "SB3")
        cs2 = _safe_num(sb, "CS2")
        if not pd.isna(sb2) or not pd.isna(sb3):
            section_header("Stolen Base Breakdown")
            sb_data = []
            sb2_pct = _safe_num(sb, "SB2%")
            sb3_pct = _safe_num(sb, "SB3%")
            if not pd.isna(sb2):
                sb_data.append({"Base": "2nd", "SB": int(sb2), "CS": int(cs2) if not pd.isna(cs2) else 0,
                                "SB%": f"{sb2_pct:.1f}%" if not pd.isna(sb2_pct) else "-"})
            if not pd.isna(sb3):
                cs3 = _safe_num(sb, "CS3")
                sb_data.append({"Base": "3rd", "SB": int(sb3), "CS": int(cs3) if not pd.isna(cs3) else 0,
                                "SB%": f"{sb3_pct:.1f}%" if not pd.isna(sb3_pct) else "-"})
            if sb_data:
                st.dataframe(pd.DataFrame(sb_data), use_container_width=True, hide_index=True)

    # ── Swing Tendencies by Pitch Type ──
    has_swing_data = not h_swpct.empty or not h_swstats.empty
    if has_swing_data:
        section_header("Swing Tendencies by Pitch Type")
        st.caption("How aggressively this hitter swings at different pitch types — key for pitch sequencing")

        swing_col1, swing_col2 = st.columns(2)

        with swing_col1:
            # Swing% by pitch type from 1P Swing%.csv
            swing_rates = []
            for lbl, col, color in [
                ("Fastball", "Swing% vs Hard", "#d22d49"),
                ("Slider", "Swing% vs SL", "#f7c631"),
                ("Curveball", "Swing% vs CB", "#00d1ed"),
                ("Changeup", "Swing% vs CH", "#1dbe3a"),
            ]:
                v = _safe_num(h_swpct, col)
                if not pd.isna(v):
                    pct = _tm_pctile(h_swpct, col, all_h_swpct)
                    swing_rates.append({"Pitch": lbl, "Swing%": v, "Pctile": pct, "Color": color})

            if swing_rates:
                st.markdown("**Swing Rate by Pitch Type**")
                # Bar chart
                sr_df = pd.DataFrame(swing_rates)
                fig_sw = go.Figure()
                fig_sw.add_trace(go.Bar(
                    x=sr_df["Pitch"], y=sr_df["Swing%"],
                    marker_color=sr_df["Color"],
                    text=sr_df["Swing%"].apply(lambda x: f"{x:.1f}%"),
                    textposition="outside", textfont=dict(size=11, color="#1a1a2e"),
                ))
                fig_sw.update_layout(
                    **CHART_LAYOUT, height=280,
                    yaxis=dict(title="Swing %", range=[0, max(sr_df["Swing%"].max() * 1.2, 60)]),
                    xaxis=dict(title=""),
                    showlegend=False,
                )
                st.plotly_chart(fig_sw, use_container_width=True)

                # Detail table with percentiles
                tbl_data = []
                for r in swing_rates:
                    pct_str = f"{int(r['Pctile'])}th" if not pd.isna(r["Pctile"]) else "-"
                    tbl_data.append({
                        "Pitch": r["Pitch"],
                        "Swing%": f"{r['Swing%']:.1f}%",
                        "D1 %ile": pct_str,
                    })
                st.dataframe(pd.DataFrame(tbl_data), use_container_width=True, hide_index=True)

                # Narrative: identify vulnerability
                sorted_rates = sorted(swing_rates, key=lambda x: x["Swing%"], reverse=True)
                highest = sorted_rates[0]
                lowest = sorted_rates[-1]
                if highest["Swing%"] - lowest["Swing%"] > 10:
                    st.caption(
                        f"⚡ Swings most aggressively at **{highest['Pitch']}** ({highest['Swing%']:.1f}%) "
                        f"and least at **{lowest['Pitch']}** ({lowest['Swing%']:.1f}%). "
                        f"Tunnel {highest['Pitch'].lower()} look early, then use "
                        f"{lowest['Pitch'].lower()} to steal strikes."
                    )

        with swing_col2:
            # First-pitch swing tendencies
            fp_data = []
            fp_hard = _safe_num(h_swpct, "1PSwing% vs Hard Empty")
            if pd.isna(fp_hard):
                fp_hard = _safe_num(h_swstats, "1PSwing% vs Hard Empty")
            fp_ch = _safe_num(h_swstats, "1PSwing% vs CH Empty")

            if not pd.isna(fp_hard) or not pd.isna(fp_ch):
                st.markdown("**First-Pitch Aggressiveness**")
                fp_items = []
                if not pd.isna(fp_hard):
                    pct_fp_hard = _tm_pctile(h_swstats if not h_swstats.empty else h_swpct,
                                              "1PSwing% vs Hard Empty",
                                              all_h_swstats if not all_h_swstats.empty else all_h_swpct)
                    pct_str = f" ({int(pct_fp_hard)}th %ile)" if not pd.isna(pct_fp_hard) else ""
                    fp_items.append({"Situation": "1st Pitch vs Fastball (empty)", "Swing%": f"{fp_hard:.0f}%", "D1 %ile": pct_str.strip()})
                if not pd.isna(fp_ch):
                    pct_fp_ch = _tm_pctile(h_swstats, "1PSwing% vs CH Empty", all_h_swstats)
                    pct_str = f" ({int(pct_fp_ch)}th %ile)" if not pd.isna(pct_fp_ch) else ""
                    fp_items.append({"Situation": "1st Pitch vs Changeup (empty)", "Swing%": f"{fp_ch:.0f}%", "D1 %ile": pct_str.strip()})
                if fp_items:
                    st.dataframe(pd.DataFrame(fp_items), use_container_width=True, hide_index=True)

                # First-pitch narrative
                if not pd.isna(fp_hard):
                    if fp_hard >= 40:
                        st.caption(f"🔴 Very aggressive first-pitch hitter vs fastballs ({fp_hard:.0f}%) — start offspeed or off the plate.")
                    elif fp_hard <= 20:
                        st.caption(f"🟢 Patient first-pitch approach vs fastballs ({fp_hard:.0f}%) — can attack zone early with strikes.")
                    else:
                        st.caption(f"⚪ Moderate first-pitch approach ({fp_hard:.0f}%) — varies game to game.")

            # InZone Swing%, Chase%, Contact% from Swing stats
            if not h_swstats.empty:
                disc_from_sw = []
                for lbl, col in [("In-Zone Swing%", "InZoneSwing%"), ("Chase%", "Chase%"), ("Contact%", "Contact%")]:
                    v = _safe_num(h_swstats, col)
                    if not pd.isna(v):
                        pct = _tm_pctile(h_swstats, col, all_h_swstats)
                        pct_str = f"{int(pct)}th" if not pd.isna(pct) else "-"
                        disc_from_sw.append({"Metric": lbl, "Value": f"{v:.1f}%", "D1 %ile": pct_str})
                if disc_from_sw:
                    st.markdown("**Zone Discipline (Swing Stats)**")
                    st.dataframe(pd.DataFrame(disc_from_sw), use_container_width=True, hide_index=True)

    # ── 2-Strike Whiff Profile ──
    if not h_swstats.empty:
        has_whiff = any(not pd.isna(_safe_num(h_swstats, c)) for c in
                        ["2K Whiff vs LHP Hard", "2K Whiff vs LHP OS", "2K Whiff vs RHP Hard", "2K Whiff vs RHP OS"])
        if has_whiff:
            section_header("2-Strike Whiff Profile")
            st.caption("Whiff rates in 2-strike counts — the key to putting this hitter away")

            whiff_col1, whiff_col2 = st.columns(2)

            with whiff_col1:
                st.markdown("**vs LHP (2 Strikes)**")
                lhp_data = []
                for lbl, col, color in [
                    ("Hard (FB/Sinker)", "2K Whiff vs LHP Hard", "#d22d49"),
                    ("Offspeed (SL/CB/CH)", "2K Whiff vs LHP OS", "#1dbe3a"),
                ]:
                    v = _safe_num(h_swstats, col)
                    if not pd.isna(v):
                        pct = _tm_pctile(h_swstats, col, all_h_swstats)
                        pct_str = f"{int(pct)}th" if not pd.isna(pct) else "-"
                        lhp_data.append({"Pitch Type": lbl, "Whiff%": f"{v:.1f}%", "D1 %ile": pct_str, "val": v, "color": color})
                if lhp_data:
                    st.dataframe(pd.DataFrame(lhp_data)[["Pitch Type", "Whiff%", "D1 %ile"]],
                                 use_container_width=True, hide_index=True)

            with whiff_col2:
                st.markdown("**vs RHP (2 Strikes)**")
                rhp_data = []
                for lbl, col, color in [
                    ("Hard (FB/Sinker)", "2K Whiff vs RHP Hard", "#d22d49"),
                    ("Offspeed (SL/CB/CH)", "2K Whiff vs RHP OS", "#1dbe3a"),
                ]:
                    v = _safe_num(h_swstats, col)
                    if not pd.isna(v):
                        pct = _tm_pctile(h_swstats, col, all_h_swstats)
                        pct_str = f"{int(pct)}th" if not pd.isna(pct) else "-"
                        rhp_data.append({"Pitch Type": lbl, "Whiff%": f"{v:.1f}%", "D1 %ile": pct_str, "val": v, "color": color})
                if rhp_data:
                    st.dataframe(pd.DataFrame(rhp_data)[["Pitch Type", "Whiff%", "D1 %ile"]],
                                 use_container_width=True, hide_index=True)

            # Combined whiff bar chart
            all_whiff = []
            for lbl, col, color in [
                ("vs LHP Hard", "2K Whiff vs LHP Hard", "#d22d49"),
                ("vs LHP OS", "2K Whiff vs LHP OS", "#e65730"),
                ("vs RHP Hard", "2K Whiff vs RHP Hard", "#3d7dab"),
                ("vs RHP OS", "2K Whiff vs RHP OS", "#14365d"),
            ]:
                v = _safe_num(h_swstats, col)
                if not pd.isna(v):
                    all_whiff.append({"Matchup": lbl, "Whiff%": v, "Color": color})

            if len(all_whiff) >= 2:
                aw_df = pd.DataFrame(all_whiff)
                fig_wh = go.Figure()
                fig_wh.add_trace(go.Bar(
                    x=aw_df["Matchup"], y=aw_df["Whiff%"],
                    marker_color=aw_df["Color"],
                    text=aw_df["Whiff%"].apply(lambda x: f"{x:.1f}%"),
                    textposition="outside", textfont=dict(size=11, color="#1a1a2e"),
                ))
                fig_wh.update_layout(
                    **CHART_LAYOUT, height=280,
                    yaxis=dict(title="Whiff %", range=[0, max(aw_df["Whiff%"].max() * 1.3, 40)]),
                    xaxis=dict(title=""),
                    showlegend=False,
                )
                st.plotly_chart(fig_wh, use_container_width=True)

            # Whiff narrative — identify the putaway pitch
            whiff_vals = {}
            for lbl, col in [("LHP Hard", "2K Whiff vs LHP Hard"), ("LHP OS", "2K Whiff vs LHP OS"),
                              ("RHP Hard", "2K Whiff vs RHP Hard"), ("RHP OS", "2K Whiff vs RHP OS")]:
                v = _safe_num(h_swstats, col)
                if not pd.isna(v):
                    whiff_vals[lbl] = v

            if whiff_vals:
                max_whiff = max(whiff_vals, key=whiff_vals.get)
                min_whiff = min(whiff_vals, key=whiff_vals.get)
                max_v = whiff_vals[max_whiff]
                min_v = whiff_vals[min_whiff]

                putaway_notes = []
                if max_v >= 30:
                    putaway_notes.append(
                        f"🔴 **Highest whiff vulnerability: {max_whiff}** ({max_v:.1f}%) — this is the putaway sequence."
                    )
                if min_v <= 12:
                    putaway_notes.append(
                        f"🟢 **Hardest to strike out with: {min_whiff}** ({min_v:.1f}%) — avoid relying on this to finish ABs."
                    )
                # Hard vs offspeed comparison
                rhp_hard = whiff_vals.get("RHP Hard")
                rhp_os = whiff_vals.get("RHP OS")
                if rhp_hard is not None and rhp_os is not None:
                    if rhp_os > rhp_hard + 10:
                        putaway_notes.append(
                            f"⚡ vs RHP: Much more vulnerable to offspeed ({rhp_os:.1f}%) than hard stuff ({rhp_hard:.1f}%) — "
                            f"establish fastball early, finish with breaking ball."
                        )
                    elif rhp_hard > rhp_os + 10:
                        putaway_notes.append(
                            f"⚡ vs RHP: More vulnerable to hard stuff ({rhp_hard:.1f}%) than offspeed ({rhp_os:.1f}%) — "
                            f"elevate fastball for the strikeout."
                        )
                lhp_hard = whiff_vals.get("LHP Hard")
                lhp_os = whiff_vals.get("LHP OS")
                if lhp_hard is not None and lhp_os is not None:
                    if lhp_os > lhp_hard + 10:
                        putaway_notes.append(
                            f"⚡ vs LHP: More vulnerable to offspeed ({lhp_os:.1f}%) than hard ({lhp_hard:.1f}%)."
                        )
                    elif lhp_hard > lhp_os + 10:
                        putaway_notes.append(
                            f"⚡ vs LHP: More vulnerable to hard stuff ({lhp_hard:.1f}%) than offspeed ({lhp_os:.1f}%)."
                        )
                for note in putaway_notes:
                    st.markdown(note)

    # ── Platoon Splits ──
    if not h_swstats.empty:
        woba_lhp = _safe_num(h_swstats, "wOBA LHP")
        woba_rhp = _safe_num(h_swstats, "wOBA RHP")
        if not pd.isna(woba_lhp) or not pd.isna(woba_rhp):
            section_header("Platoon Splits")
            split_col1, split_col2 = st.columns(2)

            # 2K whiff values per side
            whiff_lhp_hard = _safe_num(h_swstats, "2K Whiff vs LHP Hard")
            whiff_lhp_os = _safe_num(h_swstats, "2K Whiff vs LHP OS")
            whiff_rhp_hard = _safe_num(h_swstats, "2K Whiff vs RHP Hard")
            whiff_rhp_os = _safe_num(h_swstats, "2K Whiff vs RHP OS")

            with split_col1:
                st.markdown("**vs LHP**")
                if not pd.isna(woba_lhp):
                    pct_lhp = _tm_pctile(h_swstats, "wOBA LHP", all_h_swstats)
                    pct_str = f" ({int(pct_lhp)}th %ile)" if not pd.isna(pct_lhp) else ""
                    st.metric("wOBA vs LHP", f"{woba_lhp:.3f}", help=f"D1 percentile{pct_str}")
                whiff_parts_lhp = []
                if not pd.isna(whiff_lhp_hard):
                    whiff_parts_lhp.append(f"Hard whiff **{whiff_lhp_hard:.1f}%**")
                if not pd.isna(whiff_lhp_os):
                    whiff_parts_lhp.append(f"OS whiff **{whiff_lhp_os:.1f}%**")
                if whiff_parts_lhp:
                    st.caption("2K: " + ", ".join(whiff_parts_lhp))

            with split_col2:
                st.markdown("**vs RHP**")
                if not pd.isna(woba_rhp):
                    pct_rhp = _tm_pctile(h_swstats, "wOBA RHP", all_h_swstats)
                    pct_str = f" ({int(pct_rhp)}th %ile)" if not pd.isna(pct_rhp) else ""
                    st.metric("wOBA vs RHP", f"{woba_rhp:.3f}", help=f"D1 percentile{pct_str}")
                whiff_parts_rhp = []
                if not pd.isna(whiff_rhp_hard):
                    whiff_parts_rhp.append(f"Hard whiff **{whiff_rhp_hard:.1f}%**")
                if not pd.isna(whiff_rhp_os):
                    whiff_parts_rhp.append(f"OS whiff **{whiff_rhp_os:.1f}%**")
                if whiff_parts_rhp:
                    st.caption("2K: " + ", ".join(whiff_parts_rhp))

            # Actionable narrative
            narratives = []
            if not pd.isna(woba_lhp) and not pd.isna(woba_rhp):
                diff = woba_lhp - woba_rhp
                if abs(diff) >= 0.030:
                    better_side = "LHP" if diff > 0 else "RHP"
                    worse_side = "RHP" if diff > 0 else "LHP"
                    narratives.append(
                        f"⚠️ Significantly better vs {better_side} "
                        f"(.{int(max(woba_lhp, woba_rhp)*1000):03d} vs .{int(min(woba_lhp, woba_rhp)*1000):03d}). "
                        f"Opponents should use {worse_side} arms when possible."
                    )
                elif abs(diff) < 0.020:
                    narratives.append("✅ No significant platoon split — equally effective from both sides.")

            # Whiff vulnerability narratives
            for side, hard_val, os_val, opp_hard, opp_os in [
                ("LHP", whiff_lhp_hard, whiff_lhp_os, whiff_rhp_hard, whiff_rhp_os),
                ("RHP", whiff_rhp_hard, whiff_rhp_os, whiff_lhp_hard, whiff_lhp_os),
            ]:
                if not pd.isna(hard_val) and not pd.isna(opp_hard) and hard_val >= opp_hard + 10:
                    narratives.append(
                        f"⚡ Vulnerable to hard stuff from {side} with 2 strikes ({hard_val:.0f}% whiff vs {opp_hard:.0f}%)."
                    )
                if not pd.isna(os_val) and not pd.isna(opp_os) and os_val >= opp_os + 10:
                    narratives.append(
                        f"⚡ Vulnerable to offspeed from {side} with 2 strikes ({os_val:.0f}% whiff vs {opp_os:.0f}%)."
                    )

            for note in narratives:
                st.caption(note)

    # ── How to Attack ──
    section_header("How to Attack")
    notes = _hitter_attack_plan(rate, exit_d, pr, ht, hl)

    # Enhance attack plan with swing data
    if not h_swpct.empty:
        sw_hard = _safe_num(h_swpct, "Swing% vs Hard")
        sw_sl = _safe_num(h_swpct, "Swing% vs SL")
        sw_cb = _safe_num(h_swpct, "Swing% vs CB")
        sw_ch = _safe_num(h_swpct, "Swing% vs CH")
        if not pd.isna(sw_hard) and not pd.isna(sw_cb):
            if sw_cb < sw_hard - 15:
                notes.append(f"Low curveball swing rate ({sw_cb:.1f}% vs {sw_hard:.1f}% FB) — use curves to steal strikes, set up fastball.")
        if not pd.isna(sw_ch) and not pd.isna(sw_hard):
            if sw_ch >= sw_hard:
                notes.append(f"Chases changeups at high rate ({sw_ch:.1f}%) — use CH as putaway pitch.")

    if not h_swstats.empty:
        fp_sw = _safe_num(h_swstats, "1PSwing% vs Hard Empty")
        if pd.isna(fp_sw):
            fp_sw = _safe_num(h_swpct, "1PSwing% vs Hard Empty")
        if not pd.isna(fp_sw) and fp_sw >= 40:
            notes.append(f"Aggressive first-pitch swinger ({fp_sw:.0f}%) — start offspeed or off the plate to steal strike one.")
        elif not pd.isna(fp_sw) and fp_sw <= 18:
            notes.append(f"Takes first pitch often ({fp_sw:.0f}%) — attack the zone early with a strike.")

        # 2K whiff recommendations
        rhp_hard = _safe_num(h_swstats, "2K Whiff vs RHP Hard")
        rhp_os = _safe_num(h_swstats, "2K Whiff vs RHP OS")
        if not pd.isna(rhp_hard) and not pd.isna(rhp_os):
            if rhp_os > rhp_hard and rhp_os > 25:
                notes.append(f"With 2 strikes (vs RHP): finish with offspeed ({rhp_os:.1f}% whiff) — breaking ball is the out-pitch.")
            elif rhp_hard > rhp_os and rhp_hard > 25:
                notes.append(f"With 2 strikes (vs RHP): finish with hard stuff ({rhp_hard:.1f}% whiff) — elevated fastball to finish.")

    for n in notes:
        st.markdown(f"- {n}")

    # ── Trackman Overlay ──
    _trackman_hitter_overlay(trackman_data, hitter)


def _trackman_hitter_overlay(data, hitter_name):
    """Show Trackman swing heatmap if we have pitch-level data for this hitter."""
    # Match by last name
    last_name = hitter_name.split()[-1] if " " in hitter_name else hitter_name
    matches = data[data["Batter"].str.contains(last_name, case=False, na=False)]
    if matches.empty or len(matches) < 10:
        return
    section_header("Trackman Data Overlay")
    st.caption(f"Pitch-level data from our Trackman system ({len(matches)} pitches)")
    swings = matches[matches["PitchCall"].isin(SWING_CALLS)]
    loc = swings.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    if not loc.empty and len(loc) >= 5:
        fig = px.density_heatmap(loc, x="PlateLocSide", y="PlateLocHeight", nbinsx=12, nbinsy=12,
                                 color_continuous_scale="YlOrRd")
        add_strike_zone(fig)
        fig.update_layout(title="Swing Locations", xaxis=dict(range=[-3, 3], scaleanchor="y"),
                          yaxis=dict(range=[0, 5]),
                          height=400, coloraxis_showscale=False, **CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)


def _scouting_pitcher_report(tm, team, trackman_data):
    """Their Pitchers tab — comprehensive scouting report built for game-planning."""
    p_trad = _tm_team(tm["pitching"]["traditional"], team)
    if p_trad.empty:
        st.info("No pitching data for this team.")
        return
    pitchers = sorted(p_trad["playerFullName"].unique())
    pitcher = st.selectbox("Select Pitcher", pitchers, key="sc_pitcher")

    # ── Load all data for this pitcher ──
    trad = _tm_player(p_trad, pitcher)
    rate = _tm_player(_tm_team(tm["pitching"]["rate"], team), pitcher)
    mov = _tm_player(_tm_team(tm["pitching"]["movement"], team), pitcher)
    pt = _tm_player(_tm_team(tm["pitching"]["pitch_types"], team), pitcher)
    pr = _tm_player(_tm_team(tm["pitching"]["pitch_rates"], team), pitcher)
    exit_d = _tm_player(_tm_team(tm["pitching"]["exit"], team), pitcher)
    xrate = _tm_player(_tm_team(tm["pitching"]["expected_rate"], team), pitcher)
    ht = _tm_player(_tm_team(tm["pitching"]["hit_types"], team), pitcher)
    hl = _tm_player(_tm_team(tm["pitching"]["hit_locations"], team), pitcher)
    ploc = _tm_player(_tm_team(tm["pitching"]["pitch_locations"], team), pitcher)
    p_hr = _tm_player(_tm_team(tm["pitching"]["home_runs"], team), pitcher)
    p_ptcounts = _tm_player(_tm_team(tm["pitching"]["pitch_type_counts"], team), pitcher)
    p_cnt = _tm_player(_tm_team(tm["pitching"]["counting"], team), pitcher)

    # All D1 data for percentile context
    all_p_trad = tm["pitching"]["traditional"]
    all_p_rate = tm["pitching"]["rate"]
    all_p_mov = tm["pitching"]["movement"]
    all_p_pr = tm["pitching"]["pitch_rates"]
    all_p_exit = tm["pitching"]["exit"]
    all_p_ht = tm["pitching"]["hit_types"]
    all_p_ploc = tm["pitching"]["pitch_locations"]
    all_p_xrate = tm["pitching"]["expected_rate"]
    all_p_hr = tm["pitching"]["home_runs"]

    # ── Header ──
    throws = trad.iloc[0].get("throwsHand", "?") if not trad.empty else "?"
    g = _safe_val(trad, "G", "d")
    gs = _safe_val(trad, "GS", "d")
    w = _safe_val(trad, "W", "d")
    l_val = _safe_val(trad, "L", "d")
    ip = _safe_val(trad, "IP")
    qs = _safe_val(trad, "QS", "d")
    sv = _safe_val(trad, "SV", "d")
    st.markdown(f"### {pitcher}")
    header_parts = [f"Throws: {throws}", f"G: {g}", f"GS: {gs}", f"{w}-{l_val}", f"IP: {ip}"]
    if qs != "-" and int(qs) > 0:
        header_parts.append(f"QS: {qs}")
    if sv != "-" and int(sv) > 0:
        header_parts.append(f"SV: {sv}")
    st.caption(" | ".join(header_parts))

    # ── Pitcher Narrative ──
    narrative = _pitcher_narrative(pitcher, trad, mov, pr, ht, exit_d,
                                   all_p_trad, all_p_mov, all_p_pr)
    st.markdown(narrative)

    n_pitchers = len(all_p_trad)

    # ══════════════════════════════════════════════════════════
    # SECTION 1: PERCENTILE RANKINGS
    # ══════════════════════════════════════════════════════════
    section_header("Percentile Rankings")
    st.caption(f"vs. {n_pitchers:,} D1 pitchers")
    pitching_metrics = [
        ("ERA", _safe_num(trad, "ERA"), _tm_pctile(trad, "ERA", all_p_trad), ".2f", False),
        ("FIP", _safe_num(trad, "FIP"), _tm_pctile(trad, "FIP", all_p_trad), ".2f", False),
        ("xFIP", _safe_num(rate, "xFIP"), _tm_pctile(rate, "xFIP", all_p_rate), ".2f", False),
        ("WHIP", _safe_num(trad, "WHIP"), _tm_pctile(trad, "WHIP", all_p_trad), ".2f", False),
        ("K/9", _safe_num(trad, "K/9"), _tm_pctile(trad, "K/9", all_p_trad), ".1f", True),
        ("BB/9", _safe_num(trad, "BB/9"), _tm_pctile(trad, "BB/9", all_p_trad), ".1f", False),
        ("K/BB", _safe_num(trad, "K/BB"), _tm_pctile(trad, "K/BB", all_p_trad), ".2f", True),
        ("Velo", _safe_num(mov, "Vel"), _tm_pctile(mov, "Vel", all_p_mov), ".1f", True),
        ("Spin", _safe_num(mov, "Spin"), _tm_pctile(mov, "Spin", all_p_mov), ".0f", True),
        ("Extension", _safe_num(mov, "Extension"), _tm_pctile(mov, "Extension", all_p_mov), ".1f", True),
        ("Chase %", _safe_num(pr, "Chase%"), _tm_pctile(pr, "Chase%", all_p_pr), ".1f", True),
        ("SwStrk %", _safe_num(pr, "SwStrk%"), _tm_pctile(pr, "SwStrk%", all_p_pr), ".1f", True),
        ("EV Against", _safe_num(exit_d, "ExitVel"), _tm_pctile(exit_d, "ExitVel", all_p_exit), ".1f", False),
        ("Barrel %", _safe_num(exit_d, "Barrel%"), _tm_pctile(exit_d, "Barrel%", all_p_exit), ".1f", False),
        ("GB %", _safe_num(ht, "Ground%"), _tm_pctile(ht, "Ground%", all_p_ht), ".1f", True),
        ("wOBA Agn", _safe_num(rate, "wOBA"), _tm_pctile(rate, "wOBA", all_p_rate), ".3f", False),
        ("LOB %", _safe_num(rate, "LOB%"), _tm_pctile(rate, "LOB%", all_p_rate), ".1f", True),
        ("HR/9", _safe_num(trad, "HR/9"), _tm_pctile(trad, "HR/9", all_p_trad), ".2f", False),
    ]
    pitching_metrics = [(l, v, p, f, h) for l, v, p, f, h in pitching_metrics if not pd.isna(v)]
    render_savant_percentile_section(pitching_metrics)

    # ══════════════════════════════════════════════════════════
    # SECTION 2: ARSENAL & STUFF (side by side)
    # ══════════════════════════════════════════════════════════
    if not pt.empty or not mov.empty:
        section_header("Arsenal & Stuff")

        col_ars, col_stuff = st.columns([3, 2])

        with col_ars:
            if not pt.empty:
                pitch_cols = ["4Seam%", "Sink2Seam%", "Cutter%", "Slider%", "Curve%", "Change%", "Split%", "Sweeper%"]
                pitch_labels = ["4-Seam", "Sinker", "Cutter", "Slider", "Curve", "Change", "Splitter", "Sweeper"]
                count_cols = ["4Seam#", "Sink2Seam#", "Cutter#", "Slider#", "Curve#", "Change#", "Split#", "Sweeper#"]
                arsenal_rows = []
                for pct_col, cnt_col, lbl in zip(pitch_cols, count_cols, pitch_labels):
                    pct_v = pt.iloc[0].get(pct_col) if not pt.empty else None
                    cnt_v = p_ptcounts.iloc[0].get(cnt_col) if not p_ptcounts.empty and cnt_col in p_ptcounts.columns else None
                    if pct_v is not None and not pd.isna(pct_v) and pct_v > 0:
                        arsenal_rows.append({"Pitch": lbl, "Usage": pct_v,
                                             "Count": int(cnt_v) if cnt_v is not None and not pd.isna(cnt_v) else None})
                if arsenal_rows:
                    fig = go.Figure(go.Bar(
                        x=[r["Pitch"] for r in arsenal_rows],
                        y=[r["Usage"] for r in arsenal_rows],
                        marker_color=[PITCH_COLORS.get(r["Pitch"].replace("-", ""), "#888") for r in arsenal_rows],
                        text=[f"{r['Usage']:.1f}%" for r in arsenal_rows], textposition="outside",
                        textfont=dict(size=11, color="#1a1a2e"),
                    ))
                    fig.update_layout(**CHART_LAYOUT, height=280, yaxis_title="Usage %", showlegend=False,
                                      yaxis=dict(range=[0, max(r["Usage"] for r in arsenal_rows) * 1.3]))
                    st.plotly_chart(fig, use_container_width=True)

                    # Arsenal narrative
                    primary = arsenal_rows[0] if arsenal_rows else None
                    secondary = arsenal_rows[1] if len(arsenal_rows) > 1 else None
                    if primary and secondary:
                        st.caption(
                            f"Primary: **{primary['Pitch']}** ({primary['Usage']:.1f}%) | "
                            f"Secondary: **{secondary['Pitch']}** ({secondary['Usage']:.1f}%) | "
                            f"Arsenal: **{len(arsenal_rows)} pitches**"
                        )

        with col_stuff:
            if not mov.empty:
                st.markdown("**Stuff Profile**")
                stuff_data = []
                for lbl, col, fmt in [
                    ("Avg Velo", "Vel", ".1f"), ("Max Velo", "MxVel", ".1f"),
                    ("Velo Range", "VelRange", ".1f"),
                    ("Spin (rpm)", "Spin", ".0f"),
                    ("Extension", "Extension", ".1f"),
                    ("Eff. Velo", "EffectVel", ".1f"),
                    ("IVB", "IndVertBrk", ".1f"),
                    ("Horz Break", "HorzBrk", ".1f"),
                    ("VAA", "VertApprAngle", ".2f"),
                ]:
                    v = _safe_num(mov, col)
                    if not pd.isna(v):
                        pct = _tm_pctile(mov, col, all_p_mov)
                        pct_str = f"{int(pct)}" if not pd.isna(pct) else "-"
                        stuff_data.append({"Metric": lbl, "Value": f"{v:{fmt}}", "%ile": pct_str})
                if stuff_data:
                    st.dataframe(pd.DataFrame(stuff_data), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════
    # SECTION 3: COMMAND PROFILE
    # ══════════════════════════════════════════════════════════
    if not ploc.empty or not pr.empty:
        section_header("Command Profile")

        cmd_col1, cmd_col2 = st.columns([3, 2])

        with cmd_col1:
            # 3x3 Heatmap
            if not ploc.empty:
                high = _safe_num(ploc, "High%")
                vmid = _safe_num(ploc, "VMid%")
                low = _safe_num(ploc, "Low%")
                inside = _safe_num(ploc, "Inside%")
                hmid = _safe_num(ploc, "HMid%")
                outside = _safe_num(ploc, "Outside%")
                vert = [high, vmid, low]
                horiz = [inside, hmid, outside]
                if all(not pd.isna(v) for v in vert) and all(not pd.isna(v) for v in horiz):
                    vert_total = sum(vert)
                    horiz_total = sum(horiz)
                    z_matrix = []
                    for v_val in vert:
                        row = []
                        for h_val in horiz:
                            cell_val = (v_val / max(vert_total, 1)) * (h_val / max(horiz_total, 1)) * 100
                            row.append(round(cell_val, 1))
                        z_matrix.append(row)
                    fig_hm = go.Figure(data=go.Heatmap(
                        z=z_matrix,
                        x=["Inside", "Middle", "Outside"],
                        y=["High", "Middle", "Low"],
                        colorscale=[[0, "#f0f4f8"], [0.5, "#3d7dab"], [1, "#14365d"]],
                        showscale=False,
                        text=[[f"{v:.1f}%" for v in row] for row in z_matrix],
                        texttemplate="%{text}",
                        textfont=dict(size=14, color="white"),
                        hovertemplate="Zone: %{y} %{x}<br>Frequency: %{text}<extra></extra>",
                    ))
                    fig_hm.add_shape(type="rect", x0=-0.5, y0=-0.5, x1=2.5, y1=2.5,
                                     line=dict(color="#1a1a2e", width=3))
                    fig_hm.update_layout(
                        **CHART_LAYOUT, height=280,
                        xaxis=dict(side="bottom"), yaxis=dict(autorange="reversed"),
                    )
                    fig_hm.update_layout(margin=dict(l=60, r=10, t=10, b=40))
                    st.plotly_chart(fig_hm, use_container_width=True)

        with cmd_col2:
            # Command rates with percentiles
            cmd_data = []
            # From pitch rates
            for lbl, src, col in [
                ("In Zone %", ploc, "InZone%"), ("CompLoc %", pr, "CompLoc%"),
                ("Chase %", pr, "Chase%"), ("SwStrk %", pr, "SwStrk%"),
                ("Miss %", pr, "Miss%"), ("FPStk %", pr, "FPStk%"),
            ]:
                v = _safe_num(src, col)
                if not pd.isna(v):
                    all_src = all_p_ploc if src is ploc else all_p_pr
                    pct = _tm_pctile(src, col, all_src)
                    pct_str = f"{int(pct)}" if not pd.isna(pct) else "-"
                    cmd_data.append({"Metric": lbl, "Value": f"{v:.1f}%", "%ile": pct_str})
            if cmd_data:
                st.dataframe(pd.DataFrame(cmd_data), use_container_width=True, hide_index=True)

            # Command narrative
            inzone = _safe_num(ploc, "InZone%") if not ploc.empty else np.nan
            chase = _safe_num(pr, "Chase%") if not pr.empty else np.nan
            comploc = _safe_num(pr, "CompLoc%") if not pr.empty else np.nan
            if not pd.isna(inzone) and not pd.isna(chase):
                if inzone > 55 and chase < 22:
                    st.caption("📍 Lives in the zone but doesn't expand well — sit on strikes, attack early.")
                elif inzone < 42 and chase > 30:
                    st.caption("📍 Works off the plate with high chase rate — take pitches, force strikes.")
                elif not pd.isna(comploc) and comploc > 50:
                    st.caption("📍 Plus command — hits corners consistently, be ready to swing early.")

    # ══════════════════════════════════════════════════════════
    # SECTION 3B: PLATOON SPLITS (from Trackman data)
    # ══════════════════════════════════════════════════════════
    if trackman_data is not None and not trackman_data.empty:
        last_name_ps = pitcher.split()[-1] if " " in pitcher else pitcher
        p_tm = trackman_data[trackman_data["Pitcher"].str.contains(last_name_ps, case=False, na=False)]
        if not p_tm.empty and "BatterSide" in p_tm.columns and len(p_tm) >= 30:
            left_pitches = p_tm[p_tm["BatterSide"] == "Left"]
            right_pitches = p_tm[p_tm["BatterSide"] == "Right"]

            # Compute per-side stats
            def _platoon_side_stats(side_df):
                stats = {}
                if side_df.empty:
                    return stats
                total = len(side_df)
                swings = side_df[side_df["PitchCall"].isin(SWING_CALLS)] if "PitchCall" in side_df.columns else pd.DataFrame()
                whiffs = side_df[side_df["PitchCall"] == "StrikeSwinging"] if "PitchCall" in side_df.columns else pd.DataFrame()
                stats["n_pitches"] = total
                if len(swings) > 0:
                    stats["whiff_pct"] = len(whiffs) / len(swings) * 100
                bip = side_df[side_df["PitchCall"] == "InPlay"] if "PitchCall" in side_df.columns else pd.DataFrame()
                stats["n_bip"] = len(bip)
                if len(bip) >= 5 and "ExitSpeed" in side_df.columns:
                    ev_vals = bip["ExitSpeed"].dropna()
                    if len(ev_vals) >= 5:
                        stats["avg_ev"] = ev_vals.mean()
                # Approximate PA: count sequences ending in InPlay, StrikeSwinging (K proxy),
                # or walk-related calls
                k_calls = ["StrikeSwinging"]
                bb_calls = ["BallCalled", "HitByPitch"]
                # Simple K/BB proxy: strikeouts = pitch 3 swinging strikes approximated
                # Just use counting of terminal events
                strikeouts = len(side_df[side_df["PitchCall"] == "StrikeSwinging"])
                walks = 0  # harder to derive from pitch-level; skip BB rate
                in_play = len(bip)
                approx_pa = strikeouts + in_play
                if "BallinDirt" in side_df["PitchCall"].values:
                    approx_pa += 0  # not a terminal event
                # Use called strike 3 as additional K source
                called_k = side_df[(side_df["PitchCall"] == "StrikeCalled") & (side_df.get("Strikes", pd.Series()) == 2)] if "Strikes" in side_df.columns else pd.DataFrame()
                total_k = strikeouts + len(called_k)
                if approx_pa + len(called_k) > 0:
                    stats["k_rate"] = total_k / (approx_pa + len(called_k)) * 100
                # Pitch usage breakdown
                if "TaggedPitchType" in side_df.columns:
                    usage = side_df["TaggedPitchType"].value_counts(normalize=True) * 100
                    stats["pitch_usage"] = usage.to_dict()
                return stats

            l_stats = _platoon_side_stats(left_pitches)
            r_stats = _platoon_side_stats(right_pitches)

            has_platoon_data = (l_stats.get("n_pitches", 0) >= 15 or r_stats.get("n_pitches", 0) >= 15)
            if has_platoon_data:
                section_header("Platoon Splits")
                st.caption(f"Derived from Trackman pitch-level data ({l_stats.get('n_pitches', 0)} pitches vs LHH, "
                           f"{r_stats.get('n_pitches', 0)} pitches vs RHH)")

                ps_col1, ps_col2 = st.columns(2)

                for col_ctx, label, s_stats in [(ps_col1, "vs LHH", l_stats), (ps_col2, "vs RHH", r_stats)]:
                    with col_ctx:
                        st.markdown(f"**{label}**")
                        if s_stats.get("n_pitches", 0) < 15:
                            st.caption("Limited data — not enough pitches for reliable splits.")
                        else:
                            card_data = []
                            if "whiff_pct" in s_stats:
                                card_data.append({"Metric": "Whiff %", "Value": f"{s_stats['whiff_pct']:.1f}%"})
                            if "avg_ev" in s_stats:
                                card_data.append({"Metric": "Avg EV Against", "Value": f"{s_stats['avg_ev']:.1f} mph"})
                            if "k_rate" in s_stats:
                                card_data.append({"Metric": "K Rate (approx)", "Value": f"{s_stats['k_rate']:.1f}%"})
                            card_data.append({"Metric": "BIP", "Value": str(s_stats.get("n_bip", 0))})
                            if card_data:
                                st.dataframe(pd.DataFrame(card_data), use_container_width=True, hide_index=True)

                # Pitch usage comparison table
                l_usage = l_stats.get("pitch_usage", {})
                r_usage = r_stats.get("pitch_usage", {})
                all_pitches_set = sorted(set(list(l_usage.keys()) + list(r_usage.keys())))
                if all_pitches_set:
                    usage_rows = []
                    highlight_notes = []
                    for pt_name in all_pitches_set:
                        l_pct = l_usage.get(pt_name, 0)
                        r_pct = r_usage.get(pt_name, 0)
                        usage_rows.append({"Pitch": pt_name, "vs LHH": f"{l_pct:.1f}%", "vs RHH": f"{r_pct:.1f}%"})
                        if abs(l_pct - r_pct) >= 10:
                            heavier_side = "LHH" if l_pct > r_pct else "RHH"
                            heavier_pct = max(l_pct, r_pct)
                            lighter_pct = min(l_pct, r_pct)
                            highlight_notes.append(
                                f"Throws {pt_name} **{heavier_pct:.0f}%** vs {heavier_side} but only "
                                f"**{lighter_pct:.0f}%** vs {'RHH' if heavier_side == 'LHH' else 'LHH'}"
                            )
                    st.markdown("**Pitch Usage by Side**")
                    st.dataframe(pd.DataFrame(usage_rows), use_container_width=True, hide_index=True)
                    for hn in highlight_notes:
                        st.caption(f"📊 {hn}")

                # Actionable narrative
                plat_narratives = []
                l_ev = l_stats.get("avg_ev")
                r_ev = r_stats.get("avg_ev")
                if l_ev is not None and r_ev is not None and abs(l_ev - r_ev) >= 3:
                    harder_side = "LHH" if l_ev > r_ev else "RHH"
                    plat_narratives.append(
                        f"⚠️ Gives up harder contact vs {harder_side} "
                        f"({max(l_ev, r_ev):.1f} EV vs {min(l_ev, r_ev):.1f} EV)."
                    )
                l_whiff = l_stats.get("whiff_pct")
                r_whiff = r_stats.get("whiff_pct")
                if l_whiff is not None and r_whiff is not None and abs(l_whiff - r_whiff) >= 8:
                    more_whiff_side = "LHH" if l_whiff > r_whiff else "RHH"
                    plat_narratives.append(
                        f"⚡ More swings and misses vs {more_whiff_side} "
                        f"({max(l_whiff, r_whiff):.0f}% vs {min(l_whiff, r_whiff):.0f}%)."
                    )
                # Pitch usage narrative (top highlight)
                if highlight_notes:
                    top_note = highlight_notes[0]
                    # Extract pitch name for quick tip
                    plat_narratives.append(f"📋 {top_note} — look for it early in counts.")

                if not plat_narratives:
                    if l_ev is not None and r_ev is not None and l_whiff is not None and r_whiff is not None:
                        plat_narratives.append("✅ No significant platoon advantage — consistent from both sides.")

                for pn in plat_narratives:
                    st.caption(pn)

    # ══════════════════════════════════════════════════════════
    # SECTION 4: RESULTS AGAINST (batted ball + exit velo + expected)
    # ══════════════════════════════════════════════════════════
    has_results = not exit_d.empty or not ht.empty or not xrate.empty
    if has_results:
        section_header("Results Against")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            # Batted ball & exit velo combined
            if not exit_d.empty or not ht.empty:
                st.markdown("**Contact Quality Allowed**")
                cq_data = []
                for lbl, src, col, fmt, hib in [
                    ("Exit Velo", exit_d, "ExitVel", ".1f", False),
                    ("Barrel %", exit_d, "Barrel%", ".1f", False),
                    ("Hard Hit %", exit_d, "HardOut", ".1f", False),
                    ("Launch Angle", exit_d, "LaunchAng", ".1f", None),
                    ("GB %", ht, "Ground%", ".1f", True),
                    ("FB %", ht, "Fly%", ".1f", False),
                    ("LD %", ht, "Line%", ".1f", False),
                    ("HR/FB", p_hr, "HR/FB", ".1f", False),
                ]:
                    v = _safe_num(src, col)
                    if not pd.isna(v):
                        all_src = all_p_exit if src is exit_d else (all_p_ht if src is ht else all_p_hr)
                        pct = _tm_pctile(src, col, all_src)
                        pct_str = f"{int(pct)}" if not pd.isna(pct) else "-"
                        cq_data.append({"Metric": lbl, "Value": f"{v:{fmt}}", "%ile": pct_str})
                if cq_data:
                    st.dataframe(pd.DataFrame(cq_data), use_container_width=True, hide_index=True)

        with res_col2:
            # Expected stats + spray
            if not xrate.empty:
                st.markdown("**Expected Stats Against**")
                xr_data = []
                for lbl, col in [("BA", "AVG"), ("xAVG", "xAVG"), ("SLG", "SLG"), ("xSLG", "xSLG"),
                                  ("wOBA", "wOBA"), ("xWOBA", "xWOBA"), ("BABIP", "BABIP")]:
                    v = _safe_num(xrate, col)
                    if pd.isna(v) and not rate.empty:
                        v = _safe_num(rate, col)
                    if not pd.isna(v):
                        xr_data.append({"Stat": lbl, "Value": f"{v:.3f}"})
                if xr_data:
                    st.dataframe(pd.DataFrame(xr_data), use_container_width=True, hide_index=True)

            # Spray direction
            if not hl.empty:
                spray_data = []
                for lbl, col in [("Pull %", "HPull%"), ("Center %", "HCtr%"), ("Oppo %", "HOppFld%")]:
                    v = _safe_num(hl, col)
                    if not pd.isna(v):
                        spray_data.append({"Dir": lbl, "Rate": f"{v:.1f}%"})
                if spray_data:
                    st.markdown("**Spray Against**")
                    st.dataframe(pd.DataFrame(spray_data), use_container_width=True, hide_index=True)

        # Results narrative
        ev = _safe_num(exit_d, "ExitVel") if not exit_d.empty else np.nan
        barrel = _safe_num(exit_d, "Barrel%") if not exit_d.empty else np.nan
        gb_pct = _safe_num(ht, "Ground%") if not ht.empty else np.nan
        hr_fb = _safe_num(p_hr, "HR/FB") if not p_hr.empty else np.nan

        res_notes = []
        if not pd.isna(ev):
            ev_pct = _tm_pctile(exit_d, "ExitVel", all_p_exit)
            if not pd.isna(ev_pct) and ev_pct >= 70:
                res_notes.append(f"⚠️ Hitters making hard contact ({ev:.1f} mph EV, {int(ev_pct)}th %ile) — quality of contact is a concern.")
            elif not pd.isna(ev_pct) and ev_pct <= 30:
                res_notes.append(f"✅ Suppresses hard contact ({ev:.1f} mph EV, {int(ev_pct)}th %ile) — limits damage.")
        if not pd.isna(barrel) and barrel > 8:
            res_notes.append(f"⚠️ High barrel rate allowed ({barrel:.1f}%) — vulnerable to power hitters.")
        if not pd.isna(gb_pct) and gb_pct > 50:
            res_notes.append(f"✅ Ground-ball pitcher ({gb_pct:.1f}% GB) — elevate to beat him, hit the ball in the air.")
        elif not pd.isna(gb_pct) and gb_pct < 35:
            res_notes.append(f"Fly-ball pitcher ({gb_pct:.1f}% GB) — vulnerable to HR, power approach works.")
        if not pd.isna(hr_fb) and hr_fb > 12:
            res_notes.append(f"⚠️ Gives up HR on fly balls ({hr_fb:.1f}% HR/FB) — look to drive the ball in the air.")
        for note in res_notes:
            st.markdown(note)

    # ══════════════════════════════════════════════════════════
    # SECTION 5: HOW TO ATTACK (comprehensive game plan)
    # ══════════════════════════════════════════════════════════
    section_header("How to Attack")

    # ── Arsenal Weakness Summary ──
    st.markdown("**Arsenal Weaknesses**")
    arsenal_notes = []
    if not exit_d.empty:
        ev_ag = _safe_num(exit_d, "ExitVel")
        brl = _safe_num(exit_d, "Barrel%")
        if not pd.isna(ev_ag) and ev_ag > 88:
            arsenal_notes.append(f"High EV against ({ev_ag:.1f} mph) — gets hit hard")
        if not pd.isna(brl) and brl > 8:
            arsenal_notes.append(f"High barrel rate ({brl:.1f}%) — hittable contact")
    if not pr.empty:
        swstrk = _safe_num(pr, "SwStrk%")
        if not pd.isna(swstrk) and swstrk < 8:
            arsenal_notes.append(f"Low swing-and-miss ({swstrk:.1f}% SwStr) — can be hit")
    # Identify weak pitches (high usage + poor results)
    if not pt.empty and not exit_d.empty:
        for col, name in [("4Seam%", "Fastball"), ("Sink2Seam%", "Sinker"), ("Cutter%", "Cutter"),
                           ("Slider%", "Slider"), ("Curve%", "Curveball"), ("Change%", "Changeup"),
                           ("Sweeper%", "Sweeper")]:
            usage = pt.iloc[0].get(col)
            if usage is not None and not pd.isna(usage) and usage > 15:
                arsenal_notes.append(f"{name} ({usage:.0f}% usage) — look for it early")
                break  # just flag the primary
    if not arsenal_notes:
        arsenal_notes.append("No glaring arsenal weakness — execute at-bats")
    for n in arsenal_notes:
        st.markdown(f"- {n}")

    # ── Command Profile ──
    st.markdown("**Command Profile**")
    cmd_notes = []
    if not pr.empty:
        comploc = _safe_num(pr, "CompLoc%")
        inzone = _safe_num(pr, "InZone%")
        chase_gen = _safe_num(pr, "Chase%")
        if not pd.isna(comploc):
            if comploc < 40:
                cmd_notes.append(f"Poor command ({comploc:.1f}% CompLoc) — be patient, wait for mistakes")
            elif comploc > 50:
                cmd_notes.append(f"Strong command ({comploc:.1f}% CompLoc) — have to be ready in zone")
        if not pd.isna(inzone) and not pd.isna(comploc):
            if not pd.isna(inzone) and inzone > 50:
                cmd_notes.append(f"Challenges in zone ({inzone:.0f}% InZone) — swing at strikes, don't fall behind")
            elif not pd.isna(inzone) and inzone < 40:
                cmd_notes.append(f"Pitches around zone ({inzone:.0f}% InZone) — discipline wins, take borderline pitches")
        if not pd.isna(chase_gen) and chase_gen > 30:
            cmd_notes.append(f"Generates chase ({chase_gen:.0f}%) — stay disciplined, don't expand")
    if not cmd_notes:
        cmd_notes.append("Average command profile")
    for n in cmd_notes:
        st.markdown(f"- {n}")

    # ── Location Tendencies ──
    st.markdown("**Location Tendencies**")
    loc_notes = []
    if not ploc.empty:
        h_pct = _safe_num(ploc, "High%")
        vm_pct = _safe_num(ploc, "VMid%")
        l_pct = _safe_num(ploc, "Low%")
        i_pct = _safe_num(ploc, "Inside%")
        hm_pct = _safe_num(ploc, "HMid%")
        o_pct = _safe_num(ploc, "Outside%")
        # Vertical tendency
        vert_parts = []
        if not pd.isna(h_pct) and h_pct >= 28:
            vert_parts.append(f"up ({h_pct:.0f}% High)")
        if not pd.isna(l_pct) and l_pct >= 32:
            vert_parts.append(f"down ({l_pct:.0f}% Low)")
        if not pd.isna(vm_pct) and vm_pct >= 38:
            vert_parts.append(f"middle ({vm_pct:.0f}% VMid)")
        # Horizontal tendency
        horiz_parts = []
        if not pd.isna(i_pct) and i_pct >= 28:
            horiz_parts.append(f"inside ({i_pct:.0f}%)")
        if not pd.isna(o_pct) and o_pct >= 28:
            horiz_parts.append(f"outside ({o_pct:.0f}%)")
        if vert_parts:
            loc_notes.append(f"Lives {' and '.join(vert_parts)}")
        if horiz_parts:
            loc_notes.append(f"Tends {' and '.join(horiz_parts)}")
    if not loc_notes:
        loc_notes.append("Balanced location profile")
    for n in loc_notes:
        st.markdown(f"- {n}")

    # ── Structured Team Approach ──
    st.markdown("**Team Approach**")
    approach = []
    # 1st pitch plan
    if not pt.empty:
        primary_pct = 0
        primary_name = ""
        for col, name in [("4Seam%", "fastball"), ("Sink2Seam%", "sinker"), ("Cutter%", "cutter"),
                           ("Slider%", "slider"), ("Curve%", "curveball"), ("Change%", "changeup"),
                           ("Sweeper%", "sweeper")]:
            v = pt.iloc[0].get(col)
            if v is not None and not pd.isna(v) and v > primary_pct:
                primary_pct = v
                primary_name = name
        if primary_pct > 0:
            approach.append(f"**1st Pitch**: Expect {primary_name} ({primary_pct:.0f}%)" +
                          (" — sit on it" if primary_pct > 50 else " — be ready"))
    # When ahead
    if not pr.empty:
        swstrk_v = _safe_num(pr, "SwStrk%")
        if not pd.isna(swstrk_v) and swstrk_v < 8:
            approach.append("**When Ahead**: Be aggressive — low swing-and-miss pitcher, attack in zone")
        else:
            approach.append("**When Ahead**: Look secondary, don't chase expanding zone")
    # 2-strike
    if not mov.empty:
        eff_vel = _safe_num(mov, "EffectVel")
        vel = _safe_num(mov, "Vel")
        if not pd.isna(eff_vel) and eff_vel > 93:
            approach.append(f"**2-Strike**: Shorten up — high effective velo ({eff_vel:.1f} mph)")
        elif not pd.isna(eff_vel) and eff_vel < 86:
            approach.append(f"**2-Strike**: Stay back — low effective velo ({eff_vel:.1f} mph), protect plate")
        else:
            approach.append("**2-Strike**: Shorten up, fight off putaway pitch, don't chase")
        ext = _safe_num(mov, "Extension")
        if not pd.isna(ext) and ext > 6.5:
            approach.append(f"Long extension ({ext:.1f} ft) — ball gets on you fast")
    if not approach:
        approach.append("Quality at-bats. Execute the plan.")
    for n in approach:
        st.markdown(f"- {n}")

    # ── Keep original narrative notes ──
    notes = _pitcher_attack_plan(trad, mov, pr, ht)
    # LOB% / clutch
    if not rate.empty:
        lob = _safe_num(rate, "LOB%")
        if not pd.isna(lob) and lob < 65:
            notes.append(f"Low LOB% ({lob:.1f}%) — doesn't strand runners. Rally and keep the line moving.")
        elif not pd.isna(lob) and lob > 80:
            notes.append(f"High LOB% ({lob:.1f}%) — tough in the clutch. Score in bunches when you break through.")
    if notes:
        st.markdown("**Additional Notes**")
        for n in notes:
            st.markdown(f"- {n}")

    # ── Trackman Overlay ──
    _trackman_pitcher_overlay(trackman_data, pitcher)


def _trackman_pitcher_overlay(data, pitcher_name):
    """Show Trackman location heatmap if we have pitch-level data for this pitcher."""
    last_name = pitcher_name.split()[-1] if " " in pitcher_name else pitcher_name
    matches = data[data["Pitcher"].str.contains(last_name, case=False, na=False)]
    if matches.empty or len(matches) < 10:
        return
    section_header("Trackman Data Overlay")
    st.caption(f"Pitch-level data from our Trackman system ({len(matches)} pitches)")
    loc = matches.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    if not loc.empty and len(loc) >= 5:
        fig = px.density_heatmap(loc, x="PlateLocSide", y="PlateLocHeight", nbinsx=12, nbinsy=12,
                                 color_continuous_scale="YlOrRd")
        add_strike_zone(fig)
        fig.update_layout(title="Pitch Locations", xaxis=dict(range=[-3, 3], scaleanchor="y"),
                          yaxis=dict(range=[0, 5]),
                          height=400, coloraxis_showscale=False, **CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)


def _scouting_catcher_report(tm, team):
    """Their Catchers tab — arm, framing, defense, pickoffs, pitch calling with percentile context."""
    c_def = _tm_team(tm["catching"]["defense"], team)
    c_frm = _tm_team(tm["catching"]["framing"], team)
    c_throws = _tm_team(tm["catching"]["throws"], team)
    c_pr = _tm_team(tm["catching"]["pitch_rates"], team)
    c_pt_rates = _tm_team(tm["catching"]["pitch_types_rates"], team)
    c_pt = _tm_team(tm["catching"]["pitch_types"], team)
    c_opp = _tm_team(tm["catching"]["opposing"], team)
    c_sba2 = _tm_team(tm["catching"]["sba2_throws"], team)
    c_pk = _tm_team(tm["catching"]["pickoffs"], team)
    c_pbwp = _tm_team(tm["catching"]["pb_wp"], team)
    c_pcnts = _tm_team(tm["catching"]["pitch_counts"], team)

    all_c_def = tm["catching"]["defense"]
    all_c_frm = tm["catching"]["framing"]
    all_c_throws = tm["catching"]["throws"]
    all_c_pbwp = tm["catching"]["pb_wp"]
    all_c_pr = tm["catching"]["pitch_rates"]

    # Merge available catchers
    all_catchers = set()
    for df in [c_def, c_frm, c_throws, c_pbwp, c_pk]:
        if not df.empty and "playerFullName" in df.columns:
            all_catchers.update(df["playerFullName"].unique())
    if not all_catchers:
        st.info("No catcher data for this team.")
        return

    catchers = sorted(all_catchers)
    catcher = st.selectbox("Select Catcher", catchers, key="sc_catcher")

    cd = _tm_player(c_def, catcher)
    cf = _tm_player(c_frm, catcher)
    ct = _tm_player(c_throws, catcher)
    cr = _tm_player(c_pr, catcher)
    c_ptr = _tm_player(c_pt_rates, catcher)
    c_ptc = _tm_player(c_pt, catcher)
    c_op = _tm_player(c_opp, catcher)
    c_s2 = _tm_player(c_sba2, catcher)
    c_pick = _tm_player(c_pk, catcher)
    c_pw = _tm_player(c_pbwp, catcher)
    c_pc = _tm_player(c_pcnts, catcher)

    st.markdown(f"### {catcher}")

    # ── Catcher Narrative ──
    narrative_parts = []
    pop = _safe_num(ct, "PopTime")
    throw_spd = _safe_num(ct, "CThrowSpd")
    frm_raa = _safe_num(cf, "FrmRAA")
    cs_pct = _safe_num(cd, "CS%")

    if not pd.isna(pop):
        pop_pct = _tm_pctile(ct, "PopTime", all_c_throws)
        pop_rank = 100 - pop_pct if not pd.isna(pop_pct) else np.nan
        if pop_rank >= 75:
            narrative_parts.append(f"**{catcher} has an elite arm** ({pop:.2f}s pop time, {int(pop_rank)}th percentile).")
        elif pop_rank >= 50:
            narrative_parts.append(f"**{catcher} has an average arm** ({pop:.2f}s pop time).")
        else:
            narrative_parts.append(f"**{catcher} has a below-average arm** ({pop:.2f}s pop time) — running opportunities exist.")

    if not pd.isna(frm_raa):
        frm_pct = _tm_pctile(cf, "FrmRAA", all_c_frm)
        if not pd.isna(frm_pct):
            if frm_pct >= 75:
                narrative_parts.append(f"Elite framer ({frm_raa:+.1f} FrmRAA, {int(frm_pct)}th pctile) — expect extra called strikes on borderline pitches.")
            elif frm_pct <= 25:
                narrative_parts.append(f"Below-average framer ({frm_raa:+.1f} FrmRAA) — borderline pitches may go our way.")

    if not pd.isna(cs_pct):
        cs_rank = _tm_pctile(cd, "CS%", all_c_def)
        if not pd.isna(cs_rank):
            if cs_rank >= 70:
                narrative_parts.append(f"Strong CS% ({cs_pct:.1f}%) — be selective with steal attempts.")
            elif cs_rank <= 30:
                narrative_parts.append(f"Low CS% ({cs_pct:.1f}%) — green light to run.")

    # Add PB/WP narrative
    pbwp_raa = _safe_num(c_pw, "PBWPRAA")
    if not pd.isna(pbwp_raa):
        pbwp_pct = _tm_pctile(c_pw, "PBWPRAA", all_c_pbwp)
        if not pd.isna(pbwp_pct):
            if pbwp_pct >= 75:
                narrative_parts.append(f"Elite blocker ({pbwp_raa:+.1f} PBWPRAA) — rarely lets balls get by.")
            elif pbwp_pct <= 25:
                narrative_parts.append(f"Struggles blocking ({pbwp_raa:+.1f} PBWPRAA) — extra bases available on balls in the dirt.")

    if narrative_parts:
        st.markdown(" ".join(narrative_parts))

    # ── Percentile Rankings ──
    n_catchers = max(len(all_c_throws), len(all_c_frm), len(all_c_def))
    catcher_metrics = [
        ("Pop Time", _safe_num(ct, "PopTime"), _tm_pctile(ct, "PopTime", all_c_throws), ".2f", False),
        ("Throw Velo", _safe_num(ct, "CThrowSpd"), _tm_pctile(ct, "CThrowSpd", all_c_throws), ".1f", True),
        ("FrmRAA", _safe_num(cf, "FrmRAA"), _tm_pctile(cf, "FrmRAA", all_c_frm), ".1f", True),
        ("SLAA", _safe_num(cf, "SLAA"), _tm_pctile(cf, "SLAA", all_c_frm), ".1f", True),
        ("SL+", _safe_num(cf, "SL+"), _tm_pctile(cf, "SL+", all_c_frm), ".0f", True),
        ("CS %", _safe_num(cd, "CS%"), _tm_pctile(cd, "CS%", all_c_def), ".1f", True),
        ("FldRAA", _safe_num(cd, "FldRAA"), _tm_pctile(cd, "FldRAA", all_c_def), ".1f", True),
        ("PBWPRAA", _safe_num(c_pw, "PBWPRAA"), _tm_pctile(c_pw, "PBWPRAA", all_c_pbwp), ".1f", True),
    ]
    catcher_metrics = [(l, v, p, f, h) for l, v, p, f, h in catcher_metrics if not pd.isna(v)]
    if catcher_metrics:
        section_header("Percentile Rankings")
        st.caption(f"vs. {n_catchers:,} D1 catchers")
        render_savant_percentile_section(catcher_metrics)

    # ── Arm Details + Defense Details (side by side) ──
    col_arm, col_def = st.columns(2)
    with col_arm:
        if not ct.empty:
            section_header("Arm Details")
            arm_data = []
            for lbl, col_name, fmt in [("Exchange Time", "CExchTime", ".2f"), ("Throw Velo", "CThrowSpd", ".1f"), ("Pop Time", "PopTime", ".2f")]:
                v = _safe_num(ct, col_name)
                if not pd.isna(v):
                    arm_data.append({"Metric": lbl, "Value": f"{v:{fmt}}"})
            on_tgt = _safe_num(ct, "CThrowsOnTrgt")
            total = _safe_num(ct, "CThrows")
            if not pd.isna(on_tgt) and not pd.isna(total) and total > 0:
                arm_data.append({"Metric": "On-Target %", "Value": f"{on_tgt/total*100:.1f}%"})
            if arm_data:
                st.dataframe(pd.DataFrame(arm_data), use_container_width=True, hide_index=True)
    with col_def:
        if not cd.empty:
            section_header("Defense Details")
            def_data = []
            for lbl, col_name in [("PB", "PB"), ("WP", "WP"), ("Blocks", "CatBlock")]:
                v = _safe_num(cd, col_name)
                if not pd.isna(v):
                    def_data.append({"Metric": lbl, "Value": f"{int(v)}"})
            if def_data:
                st.dataframe(pd.DataFrame(def_data), use_container_width=True, hide_index=True)

    # ── SBA2 Throw Details ──
    if not c_s2.empty:
        section_header("SBA2 Throw Breakdown")
        sba2_data = []
        for lbl, col_name, fmt in [
            ("SB2 Allowed", "SB2", "d"), ("CS2", "CS2", "d"),
            ("SBA2 Throws", "CSBA2Throws", "d"), ("On-Target", "CSBA2ThrowsOnTrgt", "d"),
            ("Off-Target", "CSBA2ThrowsOffTrgt", "d"),
            ("SBA2 Throw Velo", "CSBA2ThrowSpd", ".1f"),
            ("SBA2 Pop Time", "PopTimeSBA2", ".2f"),
            ("SBA2 Exchange", "CSBA2ExchTime", ".2f"),
            ("SBA2 Time to Base", "CSBA2TimeToBase", ".2f"),
        ]:
            v = _safe_num(c_s2, col_name)
            if not pd.isna(v):
                val_str = f"{int(v)}" if fmt == "d" else f"{v:{fmt}}"
                sba2_data.append({"Metric": lbl, "Value": val_str})
        if sba2_data:
            st.dataframe(pd.DataFrame(sba2_data), use_container_width=True, hide_index=True)

    # ── Pickoff Activity ──
    if not c_pick.empty:
        section_header("Pickoff Activity")
        pk_data = []
        for lbl, col_name in [
            ("Total PK Attempts", "CatcherPKAtt"), ("Pickoffs", "CatcherPK"), ("PK Errors", "CatcherPKErr"),
            ("PK to 1B Att", "PK1Att"), ("PK to 1B", "PK1"), ("PK1 Err", "PK1Err"),
            ("PK to 2B Att", "PK2Att"), ("PK to 2B", "PK2"), ("PK2 Err", "PK2Err"),
            ("PK to 3B Att", "PK3Att"), ("PK to 3B", "PK3"), ("PK3 Err", "PK3Err"),
        ]:
            v = _safe_num(c_pick, col_name)
            if not pd.isna(v) and v > 0:
                pk_data.append({"Metric": lbl, "Value": f"{int(v)}"})
        if pk_data:
            st.dataframe(pd.DataFrame(pk_data), use_container_width=True, hide_index=True)

    # ── Passed Balls & Wild Pitches ──
    if not c_pw.empty:
        section_header("Passed Balls & Wild Pitches")
        pw_data = []
        for lbl, col_name, fmt in [
            ("PB", "PB", "d"), ("WP", "WP", "d"), ("PB+WP", "PBWP", "d"),
            ("xPB+WP", "xPBWP", ".1f"), ("PB+WP Above Avg", "PBWPAA", ".1f"),
            ("PBWP+", "PBWP+", ".0f"), ("PBWPRAA", "PBWPRAA", ".1f"), ("PBWPWAR", "PBWPWAR", ".1f"),
        ]:
            v = _safe_num(c_pw, col_name)
            if not pd.isna(v):
                val_str = f"{int(v)}" if fmt == "d" else f"{v:{fmt}}"
                pw_data.append({"Metric": lbl, "Value": val_str})
        if pw_data:
            st.dataframe(pd.DataFrame(pw_data), use_container_width=True, hide_index=True)

    # ── Framing Details ──
    if not cf.empty:
        section_header("Framing Details")
        frm_data = []
        for lbl, col_name, fmt in [
            ("SLAA", "SLAA", ".1f"), ("SL+", "SL+", ".0f"), ("FrmRAA", "FrmRAA", ".1f"),
            ("FrmCntRAA", "FrmCntRAA", ".1f"), ("Strikes Framed", "StrkFrmd", "d"),
            ("Balls Framed", "BallFrmd", "d"),
        ]:
            v = _safe_num(cf, col_name)
            if not pd.isna(v):
                val_str = f"{int(v)}" if fmt == "d" else f"{v:{fmt}}"
                frm_data.append({"Metric": lbl, "Value": val_str})
        if frm_data:
            st.dataframe(pd.DataFrame(frm_data), use_container_width=True, hide_index=True)

    # ── Pitch Calling / Game Management ──
    if not cr.empty:
        section_header("Pitch Calling")
        st.caption("How this catcher calls the game — pitch rates when behind the plate")
        call_metrics = [
            ("InZone %", _safe_num(cr, "InZone%"), _tm_pctile(cr, "InZone%", all_c_pr), ".1f", True),
            ("Chase %", _safe_num(cr, "Chase%"), _tm_pctile(cr, "Chase%", all_c_pr), ".1f", True),
            ("Miss %", _safe_num(cr, "Miss%"), _tm_pctile(cr, "Miss%", all_c_pr), ".1f", True),
            ("SwStrk %", _safe_num(cr, "SwStrk%"), _tm_pctile(cr, "SwStrk%", all_c_pr), ".1f", True),
            ("CompLoc %", _safe_num(cr, "CompLoc%"), _tm_pctile(cr, "CompLoc%", all_c_pr), ".1f", True),
        ]
        call_metrics = [(l, v, p, f, h) for l, v, p, f, h in call_metrics if not pd.isna(v)]
        if call_metrics:
            render_savant_percentile_section(call_metrics)

    # ── Pitch Mix Called ──
    if not c_ptr.empty:
        section_header("Pitch Mix Called")
        pitch_cols = ["4Seam%", "Sink2Seam%", "Cutter%", "Slider%", "Curve%", "Change%", "Split%", "Sweeper%"]
        pitch_labels = ["4-Seam", "Sinker", "Cutter", "Slider", "Curve", "Change", "Splitter", "Sweeper"]
        vals = []
        for col_name, lbl in zip(pitch_cols, pitch_labels):
            v = _safe_num(c_ptr, col_name) if col_name in c_ptr.columns else np.nan
            if not pd.isna(v) and v > 0:
                vals.append({"Pitch": lbl, "Usage": v})
        if vals:
            vdf = pd.DataFrame(vals)
            fig = go.Figure(go.Bar(
                x=vdf["Pitch"], y=vdf["Usage"],
                marker_color=[PITCH_COLORS.get(p.replace("-", ""), "#888") for p in vdf["Pitch"]],
                text=[f"{u:.1f}%" for u in vdf["Usage"]], textposition="outside",
            ))
            fig.update_layout(**CHART_LAYOUT, height=280, yaxis_title="Usage %", showlegend=False,
                              yaxis=dict(range=[0, vdf["Usage"].max() * 1.3]))
            st.plotly_chart(fig, use_container_width=True)

    # ── Opposing Batters ──
    if not c_op.empty:
        section_header("Opposing Batters (When Catching)")
        opp_data = []
        for lbl, col_name, fmt in [
            ("AVG Against", "AVG", ".3f"), ("OBP Against", "OBP", ".3f"),
            ("SLG Against", "SLG", ".3f"), ("OPS Against", "OPS", ".3f"),
            ("H Allowed", "H", "d"), ("HR Allowed", "HR", "d"),
            ("K", "K", "d"), ("BB", "BB", "d"), ("HBP", "HBP", "d"),
        ]:
            v = _safe_num(c_op, col_name)
            if not pd.isna(v):
                val_str = f"{int(v)}" if fmt == "d" else f"{v:{fmt}}"
                opp_data.append({"Stat": lbl, "Value": val_str})
        if opp_data:
            st.dataframe(pd.DataFrame(opp_data), use_container_width=True, hide_index=True)




# ──────────────────────────────────────────────
# PITCH DESIGN LAB
# ──────────────────────────────────────────────

def _compute_stuff_plus(data, baseline=None, baselines_dict=None):
    """Compute Stuff+ for every pitch in data.
    Model: z-score composite of velo, IVB, HB, extension, VAA, spin rate
    relative to same pitch type across the BASELINE population.
    100 = average, each 10 = 1 stdev better.

    Args:
        data: DataFrame of pitches to score
        baseline: DEPRECATED — ignored when baselines_dict is provided.
        baselines_dict: Pre-computed dict from compute_stuff_baselines().
                        If None, computes from DuckDB automatically.
    """
    df = data.dropna(subset=["RelSpeed", "TaggedPitchType"]).copy()
    if df.empty:
        return df

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
        "Sweeper":        {"RelSpeed": 1.5, "InducedVertBreak": -1.0, "HorzBreak": 2.0, "Extension": 0.8, "VertApprAngle": -1.5, "SpinRate": 0.5},
        "Splitter":       {"RelSpeed": 1.0, "InducedVertBreak": -2.0, "HorzBreak": 0.5, "Extension": 1.0, "VertApprAngle": -2.0, "SpinRate": -0.3, "VeloDiff": 1.5},
        "Knuckle Curve":  {"RelSpeed": 1.5, "InducedVertBreak": -1.5, "HorzBreak": -1.5, "Extension": -1.5, "VertApprAngle": -2.0, "SpinRate": 0.5},
    }
    default_w = {"RelSpeed": 1.0, "InducedVertBreak": 1.0, "HorzBreak": 1.0, "Extension": 1.0, "VertApprAngle": 1.0, "SpinRate": 1.0}

    stuff_scores = []
    for pt, grp in df.groupby("TaggedPitchType"):
        w = weights.get(pt, default_w)
        bstats = baseline_stats.get(pt, {})
        z_total = pd.Series(0.0, index=grp.index)
        w_total = 0.0
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
                z_total += z.fillna(0) * weight
                w_total += abs(weight)
                continue
            if col not in grp.columns or col not in bstats:
                continue
            mu, sigma = bstats[col]
            if sigma == 0 or pd.isna(sigma) or pd.isna(mu):
                continue
            vals = grp[col].astype(float)
            z = (vals - mu) / sigma
            z_total += z.fillna(0) * weight
            w_total += abs(weight)
        if w_total > 0:
            z_total = z_total / w_total
        grp = grp.copy()
        grp["StuffPlus"] = 100 + z_total * 10
        stuff_scores.append(grp)

    if stuff_scores:
        return pd.concat(stuff_scores, ignore_index=True)
    return df


@st.cache_data(show_spinner="Building tunnel population database...")
def _build_tunnel_population(_data):
    """Compute raw tunnel composites for every pitcher in the database.

    Returns dict: pair_type (e.g. 'Fastball/Slider') → sorted numpy array of
    raw tunnel scores across all pitchers.  Used for percentile grading — a
    pitcher's Fastball/Slider tunnel is ranked against *all* Fastball/Slider
    tunnels in college baseball.
    """
    pop = {}  # pair_type → [raw_score, ...]
    # Pre-filter: only pitchers with ≥50 pitches and ≥2 pitch types qualify
    # for tunnel computation (reduces 12k → ~2-3k meaningful pitchers)
    pitcher_counts = _data.groupby("Pitcher").agg(
        n=("Pitcher", "size"),
        n_types=("TaggedPitchType", "nunique"),
    )
    eligible = pitcher_counts[(pitcher_counts["n"] >= 50) & (pitcher_counts["n_types"] >= 2)].index
    for pitcher in eligible:
        pdf = _data[_data["Pitcher"] == pitcher]
        tdf = _compute_tunnel_score(pdf)  # raw mode (no tunnel_pop)
        if tdf.empty:
            continue
        for _, row in tdf.iterrows():
            pair_key = '/'.join(sorted([row["Pitch A"], row["Pitch B"]]))
            pop.setdefault(pair_key, []).append(row["Tunnel Score"])
    # Convert to sorted arrays for fast percentile lookup
    for k in pop:
        pop[k] = np.array(sorted(pop[k]))
    return pop


def _compute_tunnel_score(pdf, tunnel_pop=None):
    """Compute tunnel scores using Euler-integrated flight paths and
    data-driven percentile grading.

    V5 — Backtest-calibrated rebuild:
      1. Commit point at 280ms before plate (research-backed decision window),
         not 167ms.  Produces realistic commit separations (median ~7.7").
      2. Percentile grading relative to SAME PAIR TYPE — a Sinker/Slider pair
         is compared to other Sinker/Slider tunnels, not to Fastball/Changeup.
      3. Regression-weighted composite score from 200,000 actual consecutive
         pitch pairs.  Weights derived from logistic regression on whiff:
           commit_sep  55%  (lower → more whiffs — induces bad swings)
           plate_sep   19%  (higher → more whiffs given swing — late break)
           rel_sep     10%  (lower → better — consistent arm slot)
           rel_angle    8%  (lower → better — similar launch angles)
           move_div     6%  (minor — captured by plate_sep)
           velo_gap     2%  (negligible)
      4. Release-point variance penalty, pitch-by-pitch pairing unchanged.
    """
    # Minimum columns: need pitch type, release point, plate location, and velo.
    # IVB/HB are preferred but we can fall back to 9-param or gravity-only.
    req_base = ["TaggedPitchType", "RelHeight", "RelSide", "PlateLocHeight",
                "PlateLocSide", "RelSpeed"]
    if not all(c in pdf.columns for c in req_base):
        return pd.DataFrame()

    has_ivb = "InducedVertBreak" in pdf.columns and "HorzBreak" in pdf.columns
    has_rel_angle = "VertRelAngle" in pdf.columns and "HorzRelAngle" in pdf.columns
    has_9p = all(c in pdf.columns for c in ["x0", "y0", "z0", "vx0", "vy0", "vz0", "ax0", "ay0", "az0"])

    pitch_types = pdf["TaggedPitchType"].unique()
    if len(pitch_types) < 2:
        return pd.DataFrame()

    MOUND_DIST = 60.5   # feet, rubber to plate
    GRAVITY = 32.17      # ft/s²
    COMMIT_TIME = 0.280  # seconds before plate arrival (research-backed)
    N_STEPS = 20         # Euler integration steps

    # Pair-type benchmarks: commit_sep (p10, p25, p50, p75, p90, mean, std)
    # Built from 200,000 diff-type pairs at 280ms commit point
    # Data-driven benchmarks: computed from 200,000 consecutive pitch-by-pitch
    # pairs using the same Euler physics model (280ms commit, IVB/HB accel).
    # Calibrated at the PITCH-BY-PITCH level to match how _compute_tunnel_score
    # actually scores pitchers (not aggregate-to-aggregate).
    # Format: (p10, p25, p50, p75, p90, mean, std)
    PAIR_BENCHMARKS = {
        'Changeup/Curveball': (3.5, 5.8, 9.3, 13.5, 18.1, 10.2, 5.9),
        'Changeup/Cutter': (2.9, 4.9, 7.6, 11.1, 14.8, 8.4, 5.4),
        'Changeup/Fastball': (3.0, 5.0, 7.9, 11.2, 14.8, 8.5, 4.9),
        'Changeup/Sinker': (2.9, 4.8, 7.4, 10.8, 14.0, 8.1, 4.6),
        'Changeup/Slider': (3.2, 5.2, 8.1, 11.8, 15.8, 9.0, 5.2),
        'Changeup/Splitter': (3.4, 5.7, 8.6, 13.2, 17.1, 9.4, 5.2),
        'Changeup/Sweeper': (3.5, 5.7, 8.7, 12.4, 16.3, 9.5, 5.4),
        'Curveball/Cutter': (3.1, 5.1, 8.3, 11.9, 16.1, 9.2, 6.9),
        'Curveball/Fastball': (3.1, 5.2, 8.3, 12.2, 16.4, 9.2, 5.6),
        'Curveball/Sinker': (2.8, 4.9, 7.8, 12.1, 15.8, 8.9, 5.5),
        'Curveball/Slider': (3.2, 5.4, 8.6, 12.5, 17.1, 9.5, 5.7),
        'Curveball/Splitter': (3.8, 6.1, 9.1, 13.1, 18.5, 10.2, 5.8),
        'Curveball/Sweeper': (6.3, 7.8, 10.2, 14.9, 17.7, 11.3, 4.9),
        'Cutter/Fastball': (2.6, 4.3, 6.7, 9.6, 12.7, 7.3, 4.1),
        'Cutter/Sinker': (2.3, 4.0, 6.4, 9.4, 12.7, 7.1, 4.3),
        'Cutter/Slider': (2.8, 4.6, 7.5, 10.9, 14.8, 8.3, 5.2),
        'Cutter/Splitter': (2.7, 4.5, 7.5, 10.5, 14.3, 8.2, 4.7),
        'Cutter/Sweeper': (2.1, 5.4, 7.5, 11.4, 14.9, 8.4, 4.4),
        'Fastball/Sinker': (2.4, 4.0, 6.3, 9.3, 12.4, 7.0, 4.2),
        'Fastball/Slider': (2.9, 4.8, 7.6, 10.9, 14.6, 8.3, 4.9),
        'Fastball/Splitter': (3.0, 4.7, 7.7, 11.1, 14.9, 8.4, 4.8),
        'Fastball/Sweeper': (2.9, 5.0, 7.6, 11.0, 15.2, 8.5, 5.1),
        'Sinker/Slider': (2.7, 4.6, 7.3, 10.7, 14.4, 8.1, 4.7),
        'Sinker/Splitter': (2.8, 4.1, 7.3, 10.7, 14.3, 8.1, 5.1),
        'Sinker/Sweeper': (3.1, 5.5, 7.9, 11.7, 14.0, 8.5, 4.2),
        'Slider/Splitter': (3.2, 5.3, 8.5, 12.8, 16.6, 9.4, 5.4),
        'Slider/Sweeper': (2.8, 4.4, 7.0, 11.1, 16.3, 8.3, 5.2),
    }
    DEFAULT_BENCHMARK = (2.9, 4.9, 7.7, 11.2, 15.0, 8.5, 5.1)

    # ── Per-pitch-type aggregates (used as fallback & for diagnostics) ──
    agg_cols = {
        "rel_h": ("RelHeight", "mean"), "rel_s": ("RelSide", "mean"),
        "rel_h_std": ("RelHeight", "std"), "rel_s_std": ("RelSide", "std"),
        "loc_h": ("PlateLocHeight", "mean"), "loc_s": ("PlateLocSide", "mean"),
        "velo": ("RelSpeed", "mean"), "count": ("RelSpeed", "count"),
    }
    if has_ivb:
        agg_cols["ivb"] = ("InducedVertBreak", "mean")
        agg_cols["hb"] = ("HorzBreak", "mean")
    if "Extension" in pdf.columns:
        agg_cols["ext"] = ("Extension", "mean")
    # 9-param aggregates (for fallback trajectory)
    if has_rel_angle:
        agg_cols["vert_rel_angle"] = ("VertRelAngle", "mean")
        agg_cols["horz_rel_angle"] = ("HorzRelAngle", "mean")
    if has_9p:
        for c9 in ["x0", "z0", "vx0", "vy0", "vz0", "ax0", "ay0", "az0"]:
            agg_cols[c9] = (c9, "mean")
    agg = pdf.groupby("TaggedPitchType").agg(**agg_cols).dropna(subset=["rel_h", "velo"])
    agg = agg[agg["count"] >= 10]  # need meaningful sample per pitch type
    if len(agg) < 2:
        return pd.DataFrame()
    if "ext" not in agg.columns:
        agg["ext"] = 6.0
    if "ivb" not in agg.columns:
        agg["ivb"] = np.nan
    if "hb" not in agg.columns:
        agg["hb"] = np.nan
    # Fill NaN stds with 0 (single-pitch groups)
    agg["rel_h_std"] = agg["rel_h_std"].fillna(0)
    agg["rel_s_std"] = agg["rel_s_std"].fillna(0)

    # ── Euler-integrated flight path (Method 1: IVB/HB — 96.6% of data) ──
    def _euler_trajectory(rel_h, rel_s, loc_h, loc_s, ivb, hb, velo_mph, ext):
        """Return list of (x, y, t) at N_STEPS+1 points from release to plate.
        Uses drag + Magnus via IVB/HB decomposed into continuous accelerations."""
        ext = ext if not pd.isna(ext) else 6.0
        actual_dist = MOUND_DIST - ext
        velo_fps = velo_mph * 5280.0 / 3600.0
        if velo_fps < 50:
            velo_fps = 50.0  # safety floor
        t_total = actual_dist / velo_fps
        dt = t_total / N_STEPS

        # IVB/HB are total inches of break over the full flight.
        # Model as constant acceleration: break = 0.5 * a_break * t_total^2
        # => a_break = 2 * break_ft / t_total^2
        ivb_ft = ivb / 12.0
        hb_ft = hb / 12.0
        a_ivb = 2.0 * ivb_ft / (t_total ** 2) if t_total > 0 else 0
        a_hb = 2.0 * hb_ft / (t_total ** 2) if t_total > 0 else 0

        # Initial velocities: solve for vy0, vx0 so that endpoint = plate loc
        # y(T) = rel_h + vy0*T - 0.5*g*T^2 + 0.5*a_ivb*T^2 = loc_h
        # => vy0 = (loc_h - rel_h + 0.5*g*T^2 - 0.5*a_ivb*T^2) / T
        T = t_total
        vy0 = (loc_h - rel_h + 0.5 * GRAVITY * T**2 - 0.5 * a_ivb * T**2) / T if T > 0 else 0
        vx0 = (loc_s - rel_s - 0.5 * a_hb * T**2) / T if T > 0 else 0

        path = [(rel_s, rel_h, 0.0)]
        x, y = rel_s, rel_h
        vy, vx = vy0, vx0
        t = 0.0
        for _ in range(N_STEPS):
            vy += (-GRAVITY + a_ivb) * dt
            vx += a_hb * dt
            y += vy * dt
            x += vx * dt
            t += dt
            path.append((x, y, t))
        return path, t_total

    # ── Method 2: 9-parameter trajectory (Trackman x0..az0) ──
    def _trajectory_9param(x0_val, z0_val, vx0_val, vy0_val, vz0_val,
                           ax0_val, ay0_val, az0_val):
        """Euler integration using Trackman's 9-parameter model.
        Coordinate system: x0=toward plate, y0=50 (constant), z0=height,
        vx0=horiz side velocity, vy0=toward plate (negative), vz0=vertical velocity.
        ax0=horiz accel, ay0=drag decel (positive), az0=vert accel (gravity+Magnus).
        Output: path as (plate_side, height, t) — same format as _euler_trajectory.
        x0 is negated to match PlateLocSide convention."""
        velo_fps = abs(vy0_val) if abs(vy0_val) > 50 else 130.0
        t_total = MOUND_DIST / velo_fps  # approximate
        dt = t_total / N_STEPS
        # Convert to our convention: side = -x0 (Trackman x is opposite PlateLocSide)
        side, height = -x0_val, z0_val  # negate x for PlateLocSide convention
        v_side, v_height, v_fwd = -vx0_val, vz0_val, vy0_val
        a_side, a_height, a_fwd = -ax0_val, az0_val, ay0_val
        path = [(side, height, 0.0)]
        t = 0.0
        for _ in range(N_STEPS):
            v_side += a_side * dt
            v_height += a_height * dt
            v_fwd += a_fwd * dt
            side += v_side * dt
            height += v_height * dt
            t += dt
            path.append((side, height, t))
        return path, t_total

    # ── Method 3: Gravity-only trajectory (no break data) ──
    def _gravity_trajectory(rel_h, rel_s, loc_h, loc_s, velo_mph, ext):
        """Simple trajectory with gravity only — IVB=0, HB=0.
        Used when neither IVB/HB nor 9-param data is available."""
        return _euler_trajectory(rel_h, rel_s, loc_h, loc_s, 0.0, 0.0, velo_mph, ext)

    # ── Unified trajectory dispatcher ──
    def _compute_path(row_data):
        """Choose best available trajectory method for a pitch.
        row_data: dict-like with pitch columns.
        Returns (path, t_total) or None if data is too broken."""
        ivb_val = row_data.get("ivb", np.nan) if not isinstance(row_data, pd.Series) else row_data.get("ivb", np.nan)
        hb_val = row_data.get("hb", np.nan) if not isinstance(row_data, pd.Series) else row_data.get("hb", np.nan)
        # For individual pitches (Series), use column names directly
        if isinstance(row_data, pd.Series):
            ivb_val = row_data.get("InducedVertBreak", np.nan)
            hb_val = row_data.get("HorzBreak", np.nan)

        rel_h = row_data.get("rel_h", row_data.get("RelHeight", np.nan))
        rel_s = row_data.get("rel_s", row_data.get("RelSide", np.nan))
        loc_h = row_data.get("loc_h", row_data.get("PlateLocHeight", np.nan))
        loc_s = row_data.get("loc_s", row_data.get("PlateLocSide", np.nan))
        velo = row_data.get("velo", row_data.get("RelSpeed", np.nan))
        ext = row_data.get("ext", row_data.get("Extension", 6.0))

        if pd.isna(rel_h) or pd.isna(velo) or pd.isna(loc_h):
            return None

        # Method 1: IVB/HB Euler (best, 96.6% of data)
        if not pd.isna(ivb_val) and not pd.isna(hb_val):
            return _euler_trajectory(rel_h, rel_s, loc_h, loc_s, ivb_val, hb_val, velo, ext)

        # Method 2: 9-param model (0.8% — e.g. TedABroerStadium)
        if isinstance(row_data, pd.Series):
            x0_v = row_data.get("x0", np.nan)
            z0_v = row_data.get("z0", np.nan)
            vx0_v = row_data.get("vx0", np.nan)
            vy0_v = row_data.get("vy0", np.nan)
            vz0_v = row_data.get("vz0", np.nan)
            ax0_v = row_data.get("ax0", np.nan)
            ay0_v = row_data.get("ay0", np.nan)
            az0_v = row_data.get("az0", np.nan)
        else:
            x0_v = row_data.get("x0", np.nan)
            z0_v = row_data.get("z0", np.nan)
            vx0_v = row_data.get("vx0", np.nan)
            vy0_v = row_data.get("vy0", np.nan)
            vz0_v = row_data.get("vz0", np.nan)
            ax0_v = row_data.get("ax0", np.nan)
            ay0_v = row_data.get("ay0", np.nan)
            az0_v = row_data.get("az0", np.nan)
        if not pd.isna(x0_v) and not pd.isna(vx0_v):
            return _trajectory_9param(x0_v, z0_v, vx0_v, vy0_v, vz0_v,
                                      ax0_v, ay0_v, az0_v)

        # Method 3: Gravity-only (2.0% — indoor sessions without break data)
        if not pd.isna(rel_h) and not pd.isna(loc_h):
            return _gravity_trajectory(rel_h, rel_s, loc_h, loc_s, velo, ext)

        # Method 4: Data too broken
        return None

    def _pos_at_time(path, t_total, target_t):
        """Interpolate (x, y) at a specific time from an Euler path."""
        if target_t <= 0:
            return path[0][0], path[0][1]
        if target_t >= t_total:
            return path[-1][0], path[-1][1]
        dt = t_total / N_STEPS
        idx_f = target_t / dt
        idx = int(idx_f)
        frac = idx_f - idx
        if idx >= len(path) - 1:
            return path[-1][0], path[-1][1]
        x = path[idx][0] + frac * (path[idx + 1][0] - path[idx][0])
        y = path[idx][1] + frac * (path[idx + 1][1] - path[idx][1])
        return x, y

    # ── Build pitch-by-pitch pair data (Improvement #4) ──
    pair_scores = {}  # (typeA, typeB) -> list of per-pair raw tunnel metrics
    has_pbp = "PitchofPA" in pdf.columns and "Batter" in pdf.columns
    if has_pbp:
        pbp_req = ["TaggedPitchType", "RelHeight", "RelSide", "PlateLocHeight",
                    "PlateLocSide", "RelSpeed"]
        pbp = pdf.dropna(subset=pbp_req).copy()
        if "Extension" not in pbp.columns:
            pbp["Extension"] = 6.0
        else:
            pbp["Extension"] = pbp["Extension"].fillna(6.0)
        # Sort by batter and pitch order within PA
        sort_cols = ["Batter"]
        if "Date" in pbp.columns:
            sort_cols.append("Date")
        if "Inning" in pbp.columns:
            sort_cols.append("Inning")
        sort_cols.append("PitchofPA")
        valid_sort = [c for c in sort_cols if c in pbp.columns]
        if valid_sort:
            pbp = pbp.sort_values(valid_sort)
        # Build consecutive pairs within each PA
        if "PitchofPA" in pbp.columns:
            prev = pbp.shift(1)
            same_batter = pbp["Batter"] == prev["Batter"]
            if "Date" in pbp.columns:
                same_batter = same_batter & (pbp["Date"] == prev["Date"])
            diff_type = pbp["TaggedPitchType"] != prev["TaggedPitchType"]
            pair_mask = same_batter & diff_type
            pair_idx = pbp.index[pair_mask]
            for pidx in pair_idx:
                crow = pbp.loc[pidx]
                prow = prev.loc[pidx]
                tA = prow["TaggedPitchType"]
                tB = crow["TaggedPitchType"]
                if pd.isna(tA) or pd.isna(tB):
                    continue
                key = tuple(sorted([tA, tB]))
                if key not in pair_scores:
                    pair_scores[key] = []
                pair_scores[key].append((prow, crow))

    # ── Score each pitch-type pair ──
    def _score_single_pair_from_rows(row_a, row_b):
        """Compute raw tunnel metrics for a single pitch pair using dispatcher.
        row_a, row_b: dict-like (pd.Series or agg row) with pitch columns."""
        result_a = _compute_path(row_a)
        result_b = _compute_path(row_b)
        if result_a is None or result_b is None:
            return None
        path_a, t_total_a = result_a
        path_b, t_total_b = result_b

        # Speed-adjusted commit point (#1): 280ms before each pitch arrives
        commit_t_a = max(0, t_total_a - COMMIT_TIME)
        commit_t_b = max(0, t_total_b - COMMIT_TIME)
        cax, cay = _pos_at_time(path_a, t_total_a, commit_t_a)
        cbx, cby = _pos_at_time(path_b, t_total_b, commit_t_b)
        commit_sep = np.sqrt((cay - cby)**2 + (cax - cbx)**2) * 12  # inches

        # Release separation
        rel_h_a = row_a.get("rel_h", row_a.get("RelHeight", np.nan))
        rel_s_a = row_a.get("rel_s", row_a.get("RelSide", np.nan))
        rel_h_b = row_b.get("rel_h", row_b.get("RelHeight", np.nan))
        rel_s_b = row_b.get("rel_s", row_b.get("RelSide", np.nan))
        rel_sep = np.sqrt((rel_h_a - rel_h_b)**2 + (rel_s_a - rel_s_b)**2) * 12

        # Plate separation
        loc_h_a = row_a.get("loc_h", row_a.get("PlateLocHeight", np.nan))
        loc_s_a = row_a.get("loc_s", row_a.get("PlateLocSide", np.nan))
        loc_h_b = row_b.get("loc_h", row_b.get("PlateLocHeight", np.nan))
        loc_s_b = row_b.get("loc_s", row_b.get("PlateLocSide", np.nan))
        plate_sep = np.sqrt((loc_h_a - loc_h_b)**2 + (loc_s_a - loc_s_b)**2) * 12

        # Movement divergence
        ivb_a = row_a.get("ivb", row_a.get("InducedVertBreak", 0))
        hb_a = row_a.get("hb", row_a.get("HorzBreak", 0))
        ivb_b = row_b.get("ivb", row_b.get("InducedVertBreak", 0))
        hb_b = row_b.get("hb", row_b.get("HorzBreak", 0))
        # Treat NaN break as 0 for movement divergence calc
        ivb_a = 0 if pd.isna(ivb_a) else ivb_a
        hb_a = 0 if pd.isna(hb_a) else hb_a
        ivb_b = 0 if pd.isna(ivb_b) else ivb_b
        hb_b = 0 if pd.isna(hb_b) else hb_b
        move_div = np.sqrt((ivb_a - ivb_b)**2 + (hb_a - hb_b)**2)
        velo_a = row_a.get("velo", row_a.get("RelSpeed", 0))
        velo_b = row_b.get("velo", row_b.get("RelSpeed", 0))
        velo_gap = abs(velo_a - velo_b)

        # Release angle separation (degrees)
        vra_a = row_a.get("vert_rel_angle", row_a.get("VertRelAngle", np.nan))
        hra_a = row_a.get("horz_rel_angle", row_a.get("HorzRelAngle", np.nan))
        vra_b = row_b.get("vert_rel_angle", row_b.get("VertRelAngle", np.nan))
        hra_b = row_b.get("horz_rel_angle", row_b.get("HorzRelAngle", np.nan))
        if pd.notna(vra_a) and pd.notna(vra_b) and pd.notna(hra_a) and pd.notna(hra_b):
            rel_angle_sep = np.sqrt((vra_a - vra_b)**2 + (hra_a - hra_b)**2)
        else:
            rel_angle_sep = np.nan

        return commit_sep, rel_sep, plate_sep, move_div, velo_gap, rel_angle_sep

    rows = []
    types = list(agg.index)
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            a, b = agg.loc[types[i]], agg.loc[types[j]]
            pair_key = tuple(sorted([types[i], types[j]]))

            # Try pitch-by-pitch pairing first (#4)
            pbp_pairs = pair_scores.get(pair_key, [])
            if len(pbp_pairs) >= 8:
                # Aggregate per-pair metrics
                metrics = []
                for prow, crow in pbp_pairs:
                    try:
                        m = _score_single_pair_from_rows(prow, crow)
                        if m is not None:
                            metrics.append(m)
                    except Exception:
                        continue
                if len(metrics) >= 5:
                    arr = np.array(metrics)
                    commit_sep = float(np.median(arr[:, 0]))
                    rel_sep = float(np.median(arr[:, 1]))
                    plate_sep = float(np.median(arr[:, 2]))
                    move_div = float(np.median(arr[:, 3]))
                    velo_gap = float(np.median(arr[:, 4]))
                    rel_angle_sep = float(np.nanmedian(arr[:, 5]))
                else:
                    result = _score_single_pair_from_rows(a, b)
                    if result is None:
                        continue
                    commit_sep, rel_sep, plate_sep, move_div, velo_gap, rel_angle_sep = result
            else:
                result = _score_single_pair_from_rows(a, b)
                if result is None:
                    continue
                commit_sep, rel_sep, plate_sep, move_div, velo_gap, rel_angle_sep = result

            # Release-point variance penalty (#2)
            combined_rel_std = np.sqrt(
                (a.rel_h_std**2 + b.rel_h_std**2) / 2 +
                (a.rel_s_std**2 + b.rel_s_std**2) / 2
            ) * 12  # inches
            effective_rel_sep = rel_sep + 0.5 * combined_rel_std

            # BACKTEST-CALIBRATED TUNNEL SCORE (v6)
            # Percentile grading vs same pair type, regression-weighted composite.
            # Derived from 200,000 consecutive diff-type pairs at 280ms commit.
            #
            # Logistic regression on whiff (standardised coefficients):
            #   commit_sep  55%  (lower→more whiff — induces bad swing decisions)
            #   plate_sep   19%  (higher→more whiff given swing — late divergence)
            #   rel_sep     10%  (lower→better — consistent release)
            #   rel_angle    8%  (lower→better — similar launch angles)
            #   move_div     6%  (captured by plate_sep mostly)
            #   velo_gap     2%  (negligible)

            # Look up pair-type benchmark for percentile context
            type_a_norm = types[i]; type_b_norm = types[j]
            pair_label = '/'.join(sorted([type_a_norm, type_b_norm]))
            bm = PAIR_BENCHMARKS.get(pair_label, DEFAULT_BENCHMARK)
            bm_p10, bm_p25, bm_p50, bm_p75, bm_p90, bm_mean, bm_std = bm

            # 1. COMMIT PERCENTILE (55% weight)
            # Lower commit_sep → higher percentile (better tunnel)
            # Empirical percentile mapping using actual p10/p25/p50/p75/p90
            # benchmarks — avoids Gaussian compression from large stds.
            _anchors = [
                (bm_p90, 10), (bm_p75, 25), (bm_p50, 50),
                (bm_p25, 75), (bm_p10, 90),
            ]  # lower commit_sep → higher percentile (reversed)
            if commit_sep >= bm_p90:
                commit_pct = max(0, 10 * (1 - (commit_sep - bm_p90) / max(bm_p90, 1)))
            elif commit_sep <= bm_p10:
                commit_pct = min(100, 90 + 10 * (bm_p10 - commit_sep) / max(bm_p10, 1))
            else:
                # Linear interpolation between anchors
                commit_pct = 50.0
                for k in range(len(_anchors) - 1):
                    sep_hi, pct_lo = _anchors[k]      # worse end
                    sep_lo, pct_hi = _anchors[k + 1]   # better end
                    if sep_lo <= commit_sep <= sep_hi:
                        frac = (sep_hi - commit_sep) / (sep_hi - sep_lo) if sep_hi != sep_lo else 0.5
                        commit_pct = pct_lo + frac * (pct_hi - pct_lo)
                        break

            # 2. PLATE SEPARATION (19% weight)
            # Higher plate_sep → better (more divergence at plate)
            # Normalise: 0" → 0, 30" → 100
            plate_pct = min(100, plate_sep / 30.0 * 100)

            # 3. RELEASE CONSISTENCY (10% weight)
            # Lower effective_rel_sep → better
            rel_pct = max(0, 100 - effective_rel_sep * 12)

            # 4. RELEASE ANGLE SEPARATION (8% weight)
            # Lower rel_angle_sep → better (similar launch angles = harder to read)
            # Normalise: 0° → 100, 5° → 0. Fallback to 50 (neutral) if data missing.
            if pd.notna(rel_angle_sep):
                rel_angle_pct = max(0, min(100, (1 - rel_angle_sep / 5.0) * 100))
            else:
                rel_angle_pct = 50

            # 5. MOVEMENT DIVERGENCE (6% weight)
            # Higher move_div → better
            move_pct = min(100, move_div / 30.0 * 100)

            # 6. VELO GAP (2% weight) — slight bonus for speed differential
            velo_pct = min(100, velo_gap / 15.0 * 100)

            # Weighted composite (regression-derived weights)
            raw_tunnel = round(
                commit_pct * 0.55 +
                plate_pct * 0.19 +
                rel_pct * 0.10 +
                rel_angle_pct * 0.08 +
                move_pct * 0.06 +
                velo_pct * 0.02, 2)

            # Percentile grading vs all pitchers in the database
            if tunnel_pop is not None and pair_label in tunnel_pop:
                tunnel = round(percentileofscore(tunnel_pop[pair_label], raw_tunnel, kind='rank'), 1)
            else:
                # No population data — return raw composite (used during population building)
                tunnel = round(raw_tunnel, 1)

            # Letter grades based on percentile rank among all college tunnels
            if tunnel >= 80:
                grade = "A"
            elif tunnel >= 60:
                grade = "B"
            elif tunnel >= 40:
                grade = "C"
            elif tunnel >= 20:
                grade = "D"
            else:
                grade = "F"

            # Contextual diagnosis using pair-type benchmarks
            issues = []
            fixes = []
            vs_median = commit_sep - bm_p50

            if effective_rel_sep > 4:
                if combined_rel_std > 1.5:
                    issues.append(f"release points {rel_sep:.0f}\" apart (+ {combined_rel_std:.1f}\" scatter)")
                    fixes.append("Tighten arm slot consistency — release variance hurts deception")
                else:
                    issues.append(f"release points {rel_sep:.0f}\" apart")
                    fixes.append("Work on consistent arm slot across both pitches")
            if commit_sep > bm_p75:
                issues.append(f"{commit_sep:.0f}\" commit sep ({vs_median:+.1f}\" vs {pair_label} median)")
                if velo_gap > 8:
                    fixes.append(f"Reduce {velo_gap:.0f} mph velo gap — pitches separate too early")
                else:
                    fixes.append("Pitch trajectories diverge too early — hitter can read them")
            elif commit_sep > bm_p50:
                issues.append(f"commit sep slightly above average for {pair_label} ({vs_median:+.1f}\")")
            if plate_sep < 6:
                issues.append(f"only {plate_sep:.0f}\" apart at plate")
                fixes.append("Pitches end up too close together — need more movement contrast")
            if move_div < 5:
                issues.append(f"only {move_div:.0f}\" movement difference")
                fixes.append("Increase break differential — pitches move too similarly")
            if pd.notna(rel_angle_sep) and rel_angle_sep > 3:
                issues.append(f"{rel_angle_sep:.1f}° release angle divergence")
                fixes.append("Release angles differ too much — hitter can distinguish pitch type at release")
            if not issues:
                if tunnel >= 75:
                    diagnosis = f"Elite tunnel — {tunnel:.0f}th percentile for {pair_label}"
                elif tunnel >= 50:
                    diagnosis = f"Above-average tunnel — {tunnel:.0f}th percentile for {pair_label}"
                elif tunnel >= 25:
                    diagnosis = f"Below-average tunnel — {tunnel:.0f}th percentile for {pair_label}"
                else:
                    diagnosis = f"Poor tunnel — bottom {tunnel:.0f}% for {pair_label}"
            else:
                diagnosis = "; ".join(issues)

            n_pairs_used = len(metrics) if len(pbp_pairs) >= 8 and len(metrics) >= 5 else 0

            rows.append({
                "Pitch A": types[i], "Pitch B": types[j],
                "Grade": grade, "Tunnel Score": tunnel,
                "Release Sep (in)": round(rel_sep, 1),
                "Commit Sep (in)": round(commit_sep, 1),
                "Plate Sep (in)": round(plate_sep, 1),
                "Velo Gap (mph)": round(velo_gap, 1),
                "Move Diff (in)": round(move_div, 1),
                "Rel Angle Sep (°)": round(rel_angle_sep, 1) if pd.notna(rel_angle_sep) else None,
                "Pairs Used": n_pairs_used if n_pairs_used > 0 else "avg",
                "Diagnosis": diagnosis,
                "Fix": "; ".join(fixes) if fixes else "No changes needed",
            })
    return pd.DataFrame(rows).sort_values("Tunnel Score", ascending=False).reset_index(drop=True)


def _compute_command_plus(pdf, data=None):
    """Compute Command+ for each pitch type. Returns DataFrame with
    Pitch, Pitches, Loc Spread (ft), Zone%, Edge%, CSW%, Chase%, Command+.
    Returns empty DataFrame if insufficient data."""
    cmd_rows = []
    for pt in sorted(pdf["TaggedPitchType"].unique()):
        ptd = pdf[pdf["TaggedPitchType"] == pt].dropna(subset=["PlateLocSide", "PlateLocHeight"])
        if len(ptd) < 10:
            continue
        loc_std_h = ptd["PlateLocHeight"].std()
        loc_std_s = ptd["PlateLocSide"].std()
        loc_spread = np.sqrt(loc_std_h**2 + loc_std_s**2)
        in_zone = in_zone_mask(ptd)
        zone_pct = in_zone.mean() * 100
        edge = (
            ((ptd["PlateLocSide"].abs().between(0.5, 1.1)) |
             (ptd["PlateLocHeight"].between(1.2, 1.8)) |
             (ptd["PlateLocHeight"].between(3.2, 3.8))) &
            (ptd["PlateLocSide"].abs() <= 1.5) &
            ptd["PlateLocHeight"].between(0.5, 4.5)
        )
        edge_pct = edge.mean() * 100
        csw = ptd["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"]).mean() * 100
        out_zone = ~in_zone
        chase_swings = ptd[out_zone & ptd["PitchCall"].isin(SWING_CALLS)]
        chase_pct = len(chase_swings) / max(out_zone.sum(), 1) * 100
        cmd_rows.append({
            "Pitch": pt,
            "Pitches": len(ptd),
            "Loc Spread (ft)": round(loc_spread, 2),
            "Zone%": round(zone_pct, 1),
            "Edge%": round(edge_pct, 1),
            "CSW%": round(csw, 1),
            "Chase%": round(chase_pct, 1),
        })
    if not cmd_rows:
        return pd.DataFrame()
    cmd_df = pd.DataFrame(cmd_rows)
    dav_data = data if data is not None else load_davidson_data()
    all_dav = filter_davidson(dav_data, role="pitcher")
    all_dav = normalize_pitch_types(all_dav)
    cmd_scores = []
    for _, row in cmd_df.iterrows():
        pt = row["Pitch"]
        all_pt = all_dav[all_dav["TaggedPitchType"] == pt].dropna(
            subset=["PlateLocSide", "PlateLocHeight"])
        if len(all_pt) < 20:
            cmd_scores.append(100.0)
            continue
        pitcher_spreads = []
        for p, pg in all_pt.groupby("Pitcher"):
            if len(pg) < 10:
                continue
            sp = np.sqrt(pg["PlateLocHeight"].std()**2 + pg["PlateLocSide"].std()**2)
            pitcher_spreads.append(sp)
        if len(pitcher_spreads) < 3:
            cmd_scores.append(100.0)
            continue
        pctl = 100 - percentileofscore(pitcher_spreads, row["Loc Spread (ft)"], kind="rank")
        cmd_scores.append(round(100 + (pctl - 50) * 0.4, 0))
    cmd_df["Command+"] = cmd_scores
    cmd_df = cmd_df.sort_values("Command+", ascending=False)
    return cmd_df


def _compute_pitch_pair_results(pdf, data, tunnel_df=None):
    """Compute effectiveness when pitch B follows pitch A in an at-bat."""
    if pdf.empty:
        return pd.DataFrame()

    # Sort by game, at-bat, pitch number
    sort_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning", "PitchNo"] if c in pdf.columns]
    if len(sort_cols) < 2:
        return pd.DataFrame()

    pdf_s = pdf.sort_values(sort_cols).copy()
    pdf_s["PrevPitch"] = pdf_s.groupby(["GameID", "Batter", "PAofInning"])["TaggedPitchType"].shift(1)
    pdf_s = pdf_s.dropna(subset=["PrevPitch"])

    is_whiff = pdf_s["PitchCall"] == "StrikeSwinging"
    is_swing = pdf_s["PitchCall"].isin(SWING_CALLS)
    is_csw = pdf_s["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"])

    rows = []
    for (prev, curr), grp in pdf_s.groupby(["PrevPitch", "TaggedPitchType"]):
        n = len(grp)
        if n < 25:
            continue
        swings = grp[is_swing.reindex(grp.index, fill_value=False)]
        whiffs = grp[is_whiff.reindex(grp.index, fill_value=False)]
        csws = grp[is_csw.reindex(grp.index, fill_value=False)]
        batted = grp[(grp["PitchCall"] == "InPlay") & grp["ExitSpeed"].notna()]
        # Tunnel lookup for this pair
        tun_grade, tun_score = "-", np.nan
        if isinstance(tunnel_df, pd.DataFrame) and not tunnel_df.empty:
            tun_match = tunnel_df[
                ((tunnel_df["Pitch A"] == prev) & (tunnel_df["Pitch B"] == curr)) |
                ((tunnel_df["Pitch A"] == curr) & (tunnel_df["Pitch B"] == prev))
            ]
            if not tun_match.empty:
                tun_grade = tun_match.iloc[0]["Grade"]
                tun_score = tun_match.iloc[0]["Tunnel Score"]
        rows.append({
            "Setup Pitch": prev, "Follow Pitch": curr, "Count": n,
            "Whiff%": round(len(whiffs) / max(len(swings), 1) * 100, 1),
            "CSW%": round(len(csws) / n * 100, 1),
            "Avg EV": round(batted["ExitSpeed"].mean(), 1) if len(batted) > 0 else np.nan,
            "Chase%": round(
                (lambda _iz=in_zone_mask(grp): len(grp[(~_iz) & grp["PitchCall"].isin(SWING_CALLS)]) /
                max(len(grp[~_iz]), 1) * 100)(), 1),
            "Tunnel": tun_grade,
            "Tunnel Score": tun_score,
        })
    return pd.DataFrame(rows).sort_values("Whiff%", ascending=False).reset_index(drop=True)




def _pitching_lab_content(data, pitcher, season_filter, pdf, stuff_df,
                          tab_stuff, tab_tunnel, tab_seq, tab_loc,
                          tab_sim, tab_cmd):
    """Render the Pitch Design Lab tabs. Called from page_pitching()."""
    if stuff_df is None or "StuffPlus" not in stuff_df.columns:
        with tab_stuff:
            st.error("Could not compute Stuff+ scores. Not enough data for this pitcher.")
        return

    # Pre-compute tunnel data (used by both Tunnel and Sequencing tabs)
    tunnel_pop = build_tunnel_population_pop()
    tunnel_df = _compute_tunnel_score(pdf, tunnel_pop=tunnel_pop)

    # ═══════════════════════════════════════════
    # TAB 1: STUFF+ GRADES
    # ═══════════════════════════════════════════
    with tab_stuff:
        section_header("Stuff+ Overview")
        st.caption("Stuff+ measures pitch quality based on velocity, movement, extension, and approach angle. "
                   "100 = league average, 110+ = plus pitch, 120+ = elite.")

        # Summary table
        arsenal_summary = stuff_df.groupby("TaggedPitchType").agg(
            Pitches=("StuffPlus", "count"),
            StuffPlus=("StuffPlus", "mean"),
            Velo=("RelSpeed", "mean"),
            MaxVelo=("RelSpeed", "max"),
            SpinRate=("SpinRate", "mean"),
            IVB=("InducedVertBreak", "mean"),
            HB=("HorzBreak", "mean"),
            Extension=("Extension", "mean"),
            VAA=("VertApprAngle", "mean"),
        ).sort_values("StuffPlus", ascending=False)
        arsenal_summary.columns = ["Pitches", "Stuff+", "Avg Velo", "Max Velo", "Spin Rate",
                                    "IVB (in)", "HB (in)", "Ext (ft)", "VAA"]

        # Color the Stuff+ values
        def style_stuff(val):
            if val >= 120:
                return "background-color: #be0000; color: white; font-weight: bold"
            elif val >= 110:
                return "background-color: #d22d49; color: white; font-weight: bold"
            elif val >= 105:
                return "background-color: #ee7e1e; color: white; font-weight: bold"
            elif val >= 95:
                return "background-color: #9e9e9e; color: white"
            elif val >= 90:
                return "background-color: #3d7dab; color: white"
            else:
                return "background-color: #14365d; color: white"

        formatted = arsenal_summary.copy()
        for c in ["Stuff+", "Avg Velo", "Max Velo", "Spin Rate", "IVB (in)", "HB (in)", "Ext (ft)", "VAA"]:
            if c in formatted.columns:
                formatted[c] = formatted[c].apply(lambda x: f"{x:.1f}" if not pd.isna(x) else "-")
        formatted["Pitches"] = formatted["Pitches"].astype(int)
        st.dataframe(formatted, use_container_width=True)

        # Savant-style percentile bars for Stuff+
        all_stuff = _compute_stuff_plus(data)
        if "StuffPlus" in all_stuff.columns:
            section_header("Stuff+ Percentile Rankings (vs All Pitchers in Database)")
            metrics = []
            for pt in arsenal_summary.index:
                my_val = stuff_df[stuff_df["TaggedPitchType"] == pt]["StuffPlus"].mean()
                all_pt_vals = all_stuff[all_stuff["TaggedPitchType"] == pt]["StuffPlus"]
                if len(all_pt_vals) > 5:
                    pctl = percentileofscore(all_pt_vals.dropna(), my_val, kind="rank")
                    metrics.append((pt, my_val, pctl, ".0f", True))
            if metrics:
                render_savant_percentile_section(metrics)

        # Stuff+ distribution violin plot
        section_header("Stuff+ Distribution by Pitch")
        fig_dist = go.Figure()
        for pt in sorted(stuff_df["TaggedPitchType"].unique()):
            pt_vals = stuff_df[stuff_df["TaggedPitchType"] == pt]["StuffPlus"]
            color = PITCH_COLORS.get(pt, "#888")
            fig_dist.add_trace(go.Violin(
                y=pt_vals, name=pt, box_visible=True, meanline_visible=True,
                fillcolor=color, line_color=color, opacity=0.7,
            ))
        fig_dist.add_hline(y=100, line_dash="dash", line_color="#888",
                          annotation_text="League Avg (100)", annotation_position="top left")
        fig_dist.update_layout(
            showlegend=False, height=400,
            yaxis_title="Stuff+",
            **CHART_LAYOUT,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # Stuff+ over time (rolling)
        section_header("Stuff+ Trend Over Time")
        stuff_time = stuff_df.dropna(subset=["Date"]).sort_values("Date")
        if len(stuff_time) > 20:
            window = st.slider("Rolling window", 10, 50, 25, key="pdl_stuff_window")
            fig_trend = go.Figure()
            for pt in sorted(stuff_time["TaggedPitchType"].unique()):
                pt_data = stuff_time[stuff_time["TaggedPitchType"] == pt].copy()
                pt_data["StuffRolling"] = pt_data["StuffPlus"].rolling(window, min_periods=5).mean()
                color = PITCH_COLORS.get(pt, "#888")
                fig_trend.add_trace(go.Scatter(
                    x=pt_data["Date"], y=pt_data["StuffRolling"],
                    mode="lines", name=pt, line=dict(color=color, width=2),
                ))
            fig_trend.add_hline(y=100, line_dash="dash", line_color="#888")
            fig_trend.update_layout(
                height=380, xaxis_title="Date", yaxis_title="Stuff+ (rolling avg)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02), **CHART_LAYOUT,
            )
            st.plotly_chart(fig_trend, use_container_width=True)

    # ═══════════════════════════════════════════
    # TAB 2: PITCH TUNNELS
    # ═══════════════════════════════════════════
    with tab_tunnel:
        section_header("Pitch Tunnel Analysis")
        st.caption("Tunnel Score = percentile rank vs all pitchers in the database for the same pair type. "
                   "A (top 20%) → F (bottom 20%). Based on physics-modeled flight paths at 280ms commit point. "
                   "Pairs are unordered — Changeup/Fastball and Fastball/Changeup get the same tunnel grade "
                   "(tunneling measures visual deception, which is symmetric; sequence *order* effects are captured in the Sequencing tab).")

        if tunnel_df.empty:
            st.info("Need 2+ pitch types to compute tunnels.")
        else:
            # Grade color badges
            grade_colors = {"A": "#22c55e", "B": "#3b82f6", "C": "#f59e0b", "D": "#f97316", "F": "#ef4444"}

            # Summary grade cards at top
            st.markdown("#### Tunnel Pair Grades")
            grade_cols = st.columns(min(len(tunnel_df), 5))
            for idx, (_, row) in enumerate(tunnel_df.head(5).iterrows()):
                with grade_cols[idx]:
                    gc = grade_colors.get(row["Grade"], "#888")
                    st.markdown(
                        f'<div style="text-align:center;padding:10px;border-radius:8px;border:2px solid {gc};'
                        f'background:{gc}15;margin:2px;">'
                        f'<span style="font-size:28px;font-weight:bold;color:{gc};">{row["Grade"]}</span><br>'
                        f'<span style="font-size:13px;">{row["Pitch A"]} + {row["Pitch B"]}</span><br>'
                        f'<span style="font-size:12px;color:#666;">Score: {row["Tunnel Score"]}</span>'
                        f'</div>', unsafe_allow_html=True)

            st.markdown("")

            # Detailed table (show key columns)
            display_cols = ["Pitch A", "Pitch B", "Grade", "Tunnel Score",
                            "Release Sep (in)", "Commit Sep (in)", "Plate Sep (in)",
                            "Velo Gap (mph)", "Move Diff (in)", "Rel Angle Sep (°)"]
            st.dataframe(tunnel_df[display_cols], use_container_width=True, hide_index=True)

            # Tunnel visualization — release point overlay + plate location
            section_header("Release Point Overlay")
            st.caption("Pitches that release from the same spot but end up in different locations = great tunnel")
            fig_rel = go.Figure()
            for pt in sorted(pdf["TaggedPitchType"].unique()):
                pt_data = pdf[pdf["TaggedPitchType"] == pt].dropna(subset=["RelSide", "RelHeight"])
                color = PITCH_COLORS.get(pt, "#888")
                fig_rel.add_trace(go.Scatter(
                    x=pt_data["RelSide"], y=pt_data["RelHeight"],
                    mode="markers", name=pt,
                    marker=dict(color=color, size=6, opacity=0.4),
                ))
                # Add mean crosshair
                fig_rel.add_trace(go.Scatter(
                    x=[pt_data["RelSide"].mean()], y=[pt_data["RelHeight"].mean()],
                    mode="markers", name=f"{pt} avg", showlegend=False,
                    marker=dict(color=color, size=16, symbol="x-thin", line=dict(width=3, color=color)),
                ))
            fig_rel.update_layout(
                xaxis_title="Release Side (ft)", yaxis_title="Release Height (ft)",
                height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02),
                **CHART_LAYOUT,
            )
            fig_rel.update_xaxes(scaleanchor="y", scaleratio=1)
            st.plotly_chart(fig_rel, use_container_width=True)

            # Side-by-side: tunnel pair visualization
            if len(tunnel_df) > 0:
                section_header("Best Tunnel Pair Visualization")
                best_pair = tunnel_df.iloc[0]
                pair_a, pair_b = best_pair["Pitch A"], best_pair["Pitch B"]
                st.markdown(f"**{pair_a}** + **{pair_b}** — Grade: **{best_pair['Grade']}** | Score: **{best_pair['Tunnel Score']}**")

                c1, c2 = st.columns(2)
                with c1:
                    # Movement profile of the pair
                    pair_data = pdf[pdf["TaggedPitchType"].isin([pair_a, pair_b])].dropna(
                        subset=["HorzBreak", "InducedVertBreak"])
                    fig_move = go.Figure()
                    for pt in [pair_a, pair_b]:
                        d = pair_data[pair_data["TaggedPitchType"] == pt]
                        color = PITCH_COLORS.get(pt, "#888")
                        fig_move.add_trace(go.Scatter(
                            x=d["HorzBreak"], y=d["InducedVertBreak"],
                            mode="markers", name=pt,
                            marker=dict(color=color, size=7, opacity=0.5),
                        ))
                    fig_move.update_layout(
                        title="Movement Profile", xaxis_title="Horizontal Break (in)",
                        yaxis_title="Induced Vert Break (in)", height=350, **CHART_LAYOUT,
                    )
                    fig_move.add_hline(y=0, line_color="#ccc")
                    fig_move.add_vline(x=0, line_color="#ccc")
                    st.plotly_chart(fig_move, use_container_width=True)

                with c2:
                    # Plate location of the pair
                    fig_loc = go.Figure()
                    for pt in [pair_a, pair_b]:
                        d = pdf[(pdf["TaggedPitchType"] == pt)].dropna(subset=["PlateLocSide", "PlateLocHeight"])
                        color = PITCH_COLORS.get(pt, "#888")
                        fig_loc.add_trace(go.Scatter(
                            x=d["PlateLocSide"], y=d["PlateLocHeight"],
                            mode="markers", name=pt,
                            marker=dict(color=color, size=7, opacity=0.5),
                        ))
                    add_strike_zone(fig_loc)
                    fig_loc.update_layout(
                        title="Plate Locations", xaxis_title="Plate Side (ft)",
                        yaxis_title="Plate Height (ft)", height=350,
                        xaxis=dict(range=[-2.5, 2.5], scaleanchor="y"), yaxis=dict(range=[0, 5]),
                        **CHART_LAYOUT,
                    )
                    st.plotly_chart(fig_loc, use_container_width=True)

    # ═══════════════════════════════════════════
    # TAB 3: SEQUENCING
    # ═══════════════════════════════════════════
    with tab_seq:
        section_header("Pitch Sequencing Analysis")
        st.caption("Shows what happens when Pitch B follows Pitch A in the same at-bat. "
                   "Use this to find your most effective pitch combinations.")

        pair_df = _compute_pitch_pair_results(pdf, data, tunnel_df=tunnel_df)
        if pair_df.empty:
            st.info("Not enough sequential pitch data.")
        else:
            # Sequencing effectiveness table
            st.dataframe(pair_df, use_container_width=True, hide_index=True)

            # Transition matrix heatmap
            section_header("Pitch Transition Matrix")
            st.caption("How often does each pitch follow another? Heat = frequency")
            sort_cols = [c for c in ["GameID", "Batter", "PAofInning", "PitchNo"] if c in pdf.columns]
            if len(sort_cols) >= 2:
                pdf_s = pdf.sort_values(sort_cols).copy()
                pdf_s["NextPitch"] = pdf_s.groupby(["GameID", "Batter", "PAofInning"])["TaggedPitchType"].shift(-1)
                trans = pdf_s.dropna(subset=["NextPitch"])
                if not trans.empty:
                    matrix = pd.crosstab(trans["TaggedPitchType"], trans["NextPitch"], normalize="index") * 100
                    fig_matrix = px.imshow(
                        matrix.round(1), text_auto=".0f", color_continuous_scale="RdBu_r",
                        labels=dict(x="Next Pitch", y="Current Pitch", color="Frequency %"),
                        aspect="auto",
                    )
                    fig_matrix.update_layout(height=380, **CHART_LAYOUT)
                    st.plotly_chart(fig_matrix, use_container_width=True)

            # Whiff% heatmap by sequence
            section_header("Whiff% by Pitch Sequence")
            if not pair_df.empty:
                whiff_pivot = pair_df.pivot_table(index="Setup Pitch", columns="Follow Pitch",
                                                  values="Whiff%", aggfunc="first")
                if not whiff_pivot.empty:
                    fig_whiff = px.imshow(
                        whiff_pivot.fillna(0).round(1), text_auto=".0f",
                        color_continuous_scale="YlOrRd",
                        labels=dict(x="Follow-Up Pitch", y="Setup Pitch", color="Whiff%"),
                        aspect="auto",
                    )
                    # Overlay tunnel grades on whiff heatmap
                    if "Tunnel" in pair_df.columns:
                        tunnel_pivot = pair_df.pivot_table(index="Setup Pitch", columns="Follow Pitch",
                                                            values="Tunnel", aggfunc="first")
                        tunnel_pivot = tunnel_pivot.reindex(index=whiff_pivot.index,
                                                            columns=whiff_pivot.columns)
                        custom_text = []
                        for ridx in whiff_pivot.index:
                            row_text = []
                            for cidx in whiff_pivot.columns:
                                w_val = whiff_pivot.loc[ridx, cidx]
                                t_val = tunnel_pivot.loc[ridx, cidx] if ridx in tunnel_pivot.index and cidx in tunnel_pivot.columns else "-"
                                if pd.isna(t_val):
                                    t_val = "-"
                                w_str = f"{w_val:.0f}%" if not pd.isna(w_val) and w_val > 0 else "0%"
                                row_text.append(f"{w_str} [{t_val}]")
                            custom_text.append(row_text)
                        fig_whiff.update_traces(text=custom_text, texttemplate="%{text}")
                    fig_whiff.update_layout(height=380, **CHART_LAYOUT)
                    st.plotly_chart(fig_whiff, use_container_width=True)

            # First-pitch tendencies
            section_header("Count-State Pitch Selection")
            counts_of_interest = [("0", "0"), ("0", "2"), ("1", "2"), ("2", "0"), ("3", "1"), ("3", "2")]
            count_rows = []
            for b, s in counts_of_interest:
                count_data = pdf[(pdf["Balls"].astype(str) == b) & (pdf["Strikes"].astype(str) == s)]
                if len(count_data) >= 3:
                    usage = count_data["TaggedPitchType"].value_counts(normalize=True) * 100
                    for pt, pct in usage.items():
                        count_rows.append({"Count": f"{b}-{s}", "Pitch": pt, "Usage%": round(pct, 1)})
            if count_rows:
                count_df = pd.DataFrame(count_rows)
                fig_count = px.bar(
                    count_df, x="Count", y="Usage%", color="Pitch",
                    color_discrete_map=PITCH_COLORS, barmode="stack",
                )
                fig_count.update_layout(
                    height=380, yaxis_title="Usage %", xaxis_title="Count",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    **CHART_LAYOUT,
                )
                st.plotly_chart(fig_count, use_container_width=True)

    # ═══════════════════════════════════════════
    # TAB 4: LOCATION LAB
    # ═══════════════════════════════════════════
    with tab_loc:
        section_header("Location Optimization Lab")
        st.caption("Find the exact locations where each pitch generates the most whiffs, weakest contact, and highest called strike rates.")

        pitch_type_sel = st.selectbox("Select Pitch", sorted(pdf["TaggedPitchType"].unique()), key="pdl_loc_pitch")
        pt_data = pdf[pdf["TaggedPitchType"] == pitch_type_sel].copy()

        if len(pt_data) < 10:
            st.info("Not enough pitches of this type to analyze locations.")
        else:
            c1, c2, c3 = st.columns(3)

            # Whiff heatmap
            with c1:
                whiff_data = pt_data[pt_data["PitchCall"] == "StrikeSwinging"]
                section_header("Whiff Locations")
                if len(whiff_data) >= 3:
                    fig = go.Figure(go.Histogram2d(
                        x=whiff_data["PlateLocSide"], y=whiff_data["PlateLocHeight"],
                        nbinsx=10, nbinsy=10,
                        colorscale="YlOrRd", showscale=False,
                    ))
                    add_strike_zone(fig)
                    fig.update_layout(
                        xaxis=dict(range=[-2.5, 2.5], title=""),
                        yaxis=dict(range=[0, 5], title=""),
                        height=320, **CHART_LAYOUT,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"{len(whiff_data)} whiffs")
                else:
                    st.info("Not enough whiffs")

            # Called strike heatmap
            with c2:
                cs_data = pt_data[pt_data["PitchCall"] == "StrikeCalled"]
                section_header("Called Strike Locations")
                if len(cs_data) >= 3:
                    fig = go.Figure(go.Histogram2d(
                        x=cs_data["PlateLocSide"], y=cs_data["PlateLocHeight"],
                        nbinsx=10, nbinsy=10,
                        colorscale="Blues", showscale=False,
                    ))
                    add_strike_zone(fig)
                    fig.update_layout(
                        xaxis=dict(range=[-2.5, 2.5], title=""),
                        yaxis=dict(range=[0, 5], title=""),
                        height=320, **CHART_LAYOUT,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"{len(cs_data)} called strikes")
                else:
                    st.info("Not enough called strikes")

            # Weak contact heatmap (EV < 85 on balls in play)
            with c3:
                ip_data = pt_data[(pt_data["PitchCall"] == "InPlay") & pt_data["ExitSpeed"].notna()]
                weak = ip_data[ip_data["ExitSpeed"] < 85]
                section_header("Weak Contact Locations")
                if len(weak) >= 3:
                    fig = go.Figure(go.Histogram2d(
                        x=weak["PlateLocSide"], y=weak["PlateLocHeight"],
                        nbinsx=10, nbinsy=10,
                        colorscale="Greens", showscale=False,
                    ))
                    add_strike_zone(fig)
                    fig.update_layout(
                        xaxis=dict(range=[-2.5, 2.5], title=""),
                        yaxis=dict(range=[0, 5], title=""),
                        height=320, **CHART_LAYOUT,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"{len(weak)} weak contacts (EV < 85)")
                else:
                    st.info("Not enough weak contact")

            # Zone-quadrant breakdown
            section_header("Zone Quadrant Performance")
            st.caption("Strike zone split into 9 regions — showing effectiveness in each")
            loc_data = pt_data.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
            if len(loc_data) >= 20:
                # Define 9 zones (3x3 grid within strike zone)
                h_edges = [-0.83, -0.28, 0.28, 0.83]
                v_edges = [1.5, 2.17, 2.83, 3.5]
                zone_labels = [
                    ["Down-In", "Down-Mid", "Down-Away"],
                    ["Mid-In", "Heart", "Mid-Away"],
                    ["Up-In", "Up-Mid", "Up-Away"],
                ]
                zone_rows = []
                for vi in range(3):
                    for hi in range(3):
                        mask = (
                            (loc_data["PlateLocSide"] >= h_edges[hi]) &
                            (loc_data["PlateLocSide"] < h_edges[hi + 1]) &
                            (loc_data["PlateLocHeight"] >= v_edges[vi]) &
                            (loc_data["PlateLocHeight"] < v_edges[vi + 1])
                        )
                        zone = loc_data[mask]
                        if len(zone) < 3:
                            continue
                        swings = zone[zone["PitchCall"].isin(SWING_CALLS)]
                        whiffs_z = zone[zone["PitchCall"] == "StrikeSwinging"]
                        ip_z = zone[(zone["PitchCall"] == "InPlay") & zone["ExitSpeed"].notna()]
                        zone_rows.append({
                            "Zone": zone_labels[vi][hi],
                            "Pitches": len(zone),
                            "Whiff%": round(len(whiffs_z) / max(len(swings), 1) * 100, 1),
                            "Avg EV": round(ip_z["ExitSpeed"].mean(), 1) if len(ip_z) > 0 else np.nan,
                            "Usage%": round(len(zone) / len(loc_data) * 100, 1),
                        })
                if zone_rows:
                    zone_df = pd.DataFrame(zone_rows)
                    st.dataframe(zone_df, use_container_width=True, hide_index=True)


    # ═══════════════════════════════════════════
    # TAB 5: HITTER'S EYE — PITCH FLIGHT SIMULATOR
    # ═══════════════════════════════════════════
    with tab_sim:
        section_header("Hitter's Eye — Pitch Simulator")
        st.caption("See pitches from the batter's perspective. The 3D flight path shows how each pitch "
                   "travels from the release point to the plate, and the overlay view reveals why certain "
                   "pitch pairs are so hard to distinguish.")

        sim_pitches = sorted(pdf["TaggedPitchType"].unique())
        sim_selected = st.multiselect(
            "Select pitches to overlay (pick 2+ to see tunnel effect)",
            sim_pitches, default=sim_pitches[:min(3, len(sim_pitches))],
            key="pdl_sim_pitches",
        )
        if not sim_selected:
            st.info("Select at least one pitch type above.")
        else:
            # Compute average flight path for each pitch type using physics model
            # Release point → plate (60.5 ft) with break applied as quadratic curve
            MOUND_DIST = 60.5  # ft from rubber to plate

            flight_data = {}
            for pt in sim_selected:
                ptd = pdf[pdf["TaggedPitchType"] == pt].dropna(subset=["RelSpeed"])
                if ptd.empty:
                    continue
                velo = ptd["RelSpeed"].mean()  # mph
                rel_h = ptd["RelHeight"].mean() if "RelHeight" in ptd.columns and ptd["RelHeight"].notna().any() else 5.5
                rel_s = ptd["RelSide"].mean() if "RelSide" in ptd.columns and ptd["RelSide"].notna().any() else 0.0
                ext = ptd["Extension"].mean() if "Extension" in ptd.columns and ptd["Extension"].notna().any() else 6.0
                loc_h = ptd["PlateLocHeight"].mean() if "PlateLocHeight" in ptd.columns and ptd["PlateLocHeight"].notna().any() else 2.5
                loc_s = ptd["PlateLocSide"].mean() if "PlateLocSide" in ptd.columns and ptd["PlateLocSide"].notna().any() else 0.0
                ivb = ptd["InducedVertBreak"].mean() if "InducedVertBreak" in ptd.columns and ptd["InducedVertBreak"].notna().any() else 0.0
                hb = ptd["HorzBreak"].mean() if "HorzBreak" in ptd.columns and ptd["HorzBreak"].notna().any() else 0.0

                # Flight path: 30 points from release to plate
                n_pts = 30
                actual_dist = MOUND_DIST - ext
                t_total = actual_dist / (velo * 5280 / 3600)  # seconds of flight

                # Parametric path: t goes 0→1
                ts = np.linspace(0, 1, n_pts)
                # Distance from pitcher: linear
                z_pts = ext + ts * actual_dist  # feet from rubber

                # Height: linear interpolation + quadratic gravity + IVB
                gravity_drop = 0.5 * 32.17 * (ts * t_total)**2  # feet of gravity drop
                ivb_lift = (ivb / 12.0) * ts**2  # IVB counteracts gravity (inches→feet)
                y_pts = rel_h + (loc_h - rel_h) * ts - gravity_drop + ivb_lift
                # Ensure endpoint matches plate location
                y_correction = loc_h - y_pts[-1]
                y_pts = y_pts + y_correction * ts

                # Horizontal: linear + break curve
                hb_curve = (hb / 12.0) * ts**2  # horizontal break applied quadratically
                x_pts = rel_s + (loc_s - rel_s) * ts + hb_curve
                x_correction = loc_s - x_pts[-1]
                x_pts = x_pts + x_correction * ts

                flight_data[pt] = {
                    "x": x_pts, "y": y_pts, "z": z_pts,
                    "velo": velo, "rel_h": rel_h, "rel_s": rel_s,
                    "loc_h": loc_h, "loc_s": loc_s,
                    "ivb": ivb, "hb": hb, "ext": ext,
                    "time_ms": t_total * 1000,
                }

            if flight_data:
                col_3d, col_front = st.columns(2)

                # ── 3D Flight Path (side/overhead view) ──
                with col_3d:
                    section_header("3D Pitch Flight Path")
                    fig_3d = go.Figure()
                    for pt, fd in flight_data.items():
                        color = PITCH_COLORS.get(pt, "#888")
                        # Flight path line
                        fig_3d.add_trace(go.Scatter3d(
                            x=fd["z"], y=fd["x"], z=fd["y"],
                            mode="lines+markers",
                            name=f'{pt} ({fd["velo"]:.0f} mph)',
                            line=dict(color=color, width=6),
                            marker=dict(size=2, color=color),
                        ))
                        # Release point marker
                        fig_3d.add_trace(go.Scatter3d(
                            x=[fd["z"][0]], y=[fd["x"][0]], z=[fd["y"][0]],
                            mode="markers", showlegend=False,
                            marker=dict(size=8, color=color, symbol="diamond"),
                        ))
                        # Plate arrival marker
                        fig_3d.add_trace(go.Scatter3d(
                            x=[fd["z"][-1]], y=[fd["x"][-1]], z=[fd["y"][-1]],
                            mode="markers", showlegend=False,
                            marker=dict(size=10, color=color, symbol="circle",
                                        line=dict(width=2, color="white")),
                        ))

                    # Draw strike zone at plate (z = 60.5)
                    sz_x = [-0.83, 0.83, 0.83, -0.83, -0.83]
                    sz_y = [1.5, 1.5, 3.5, 3.5, 1.5]
                    sz_z = [MOUND_DIST] * 5
                    fig_3d.add_trace(go.Scatter3d(
                        x=sz_z, y=sz_x, z=sz_y,
                        mode="lines", showlegend=False,
                        line=dict(color="#333", width=4),
                    ))
                    fig_3d.update_layout(
                        height=500,
                        scene=dict(
                            xaxis=dict(title="Distance (ft)", range=[0, 65]),
                            yaxis=dict(title="Horizontal (ft)", range=[-3, 3]),
                            zaxis=dict(title="Height (ft)", range=[0, 8]),
                            camera=dict(eye=dict(x=1.5, y=1.2, z=0.5)),
                            aspectmode="manual",
                            aspectratio=dict(x=3, y=1, z=1),
                        ),
                        margin=dict(l=0, r=0, t=30, b=0),
                        paper_bgcolor="white",
                        font=dict(size=10, color="#1a1a2e", family="Inter, Arial, sans-serif"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                    font=dict(color="#1a1a2e")),
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)

                # ── Batter's POV (front view at plate) ──
                with col_front:
                    section_header("Batter's View (at the plate)")
                    fig_front = go.Figure()

                    # Show pitch trajectories as the batter sees them
                    for pt, fd in flight_data.items():
                        color = PITCH_COLORS.get(pt, "#888")
                        # Show the last 60% of trajectory (what hitter can react to)
                        start_idx = int(len(fd["x"]) * 0.3)

                        # Increasing marker size = pitch getting closer
                        sizes = np.linspace(3, 18, len(fd["x"][start_idx:]))
                        opacities = np.linspace(0.15, 0.9, len(fd["x"][start_idx:]))

                        # Trail path
                        fig_front.add_trace(go.Scatter(
                            x=fd["x"][start_idx:], y=fd["y"][start_idx:],
                            mode="markers+lines",
                            name=f'{pt} ({fd["velo"]:.0f})',
                            marker=dict(size=sizes, color=color, opacity=0.5),
                            line=dict(color=color, width=2, dash="dot"),
                        ))
                        # Final plate location (big marker)
                        fig_front.add_trace(go.Scatter(
                            x=[fd["loc_s"]], y=[fd["loc_h"]],
                            mode="markers+text",
                            showlegend=False,
                            text=[f'{fd["velo"]:.0f}'],
                            textposition="top center",
                            textfont=dict(size=10, color=color),
                            marker=dict(size=22, color=color, opacity=0.85,
                                        line=dict(width=2, color="white")),
                        ))

                    # Strike zone
                    fig_front.add_shape(
                        type="rect", x0=-ZONE_SIDE, x1=ZONE_SIDE, y0=ZONE_HEIGHT_BOT, y1=ZONE_HEIGHT_TOP,
                        line=dict(color="#333", width=2.5),
                        fillcolor="rgba(0,0,0,0.03)",
                    )
                    # Home plate
                    fig_front.add_shape(
                        type="path",
                        path="M -0.71 0.3 L 0.71 0.3 L 0.83 0.15 L 0 0 L -0.83 0.15 Z",
                        line=dict(color="#666", width=1.5),
                        fillcolor="rgba(200,200,200,0.3)",
                    )
                    fig_front.update_layout(
                        height=500,
                        xaxis=dict(range=[-2.5, 2.5], title="Horizontal (ft)",
                                   zeroline=False, showgrid=True, gridcolor="#eee", scaleanchor="y"),
                        yaxis=dict(range=[-0.5, 5.5], title="Height (ft)",
                                   zeroline=False, showgrid=True, gridcolor="#eee"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        **CHART_LAYOUT,
                    )
                    st.plotly_chart(fig_front, use_container_width=True)

                # ── Decision Point Analysis ──
                section_header("Pitch Decision Timeline")
                st.caption("How much time does the hitter have to decide? At what point do pitches diverge?")

                timeline_rows = []
                for pt, fd in flight_data.items():
                    t_ms = fd["time_ms"]
                    # Decision point: ~167ms before plate (average human reaction time)
                    react_ms = 167
                    decision_dist = (react_ms / t_ms) * (MOUND_DIST - fd["ext"])
                    commit_dist = MOUND_DIST - decision_dist  # distance from rubber where hitter must commit

                    # At the decision point, where is the pitch?
                    frac = 1 - (react_ms / t_ms)
                    idx = min(int(frac * len(fd["x"])), len(fd["x"]) - 1)
                    dec_h = fd["y"][idx]
                    dec_s = fd["x"][idx]

                    timeline_rows.append({
                        "Pitch": pt,
                        "Velocity": f'{fd["velo"]:.0f} mph',
                        "Flight Time": f'{t_ms:.0f} ms',
                        "Decision Point": f'{commit_dist:.1f} ft from plate',
                        "Height at Decision": f'{dec_h:.2f} ft',
                        "Side at Decision": f'{dec_s:.2f} ft',
                        "React Window": f'{t_ms - react_ms:.0f} ms',
                    })
                timeline_df = pd.DataFrame(timeline_rows)
                st.dataframe(timeline_df, use_container_width=True, hide_index=True)

                # Tunnel divergence analysis
                if len(sim_selected) >= 2:
                    section_header("Tunnel Divergence Analysis")
                    st.caption("At the commit point (~280ms before plate), how far apart are the pitches? "
                               "Less separation = better deception.")

                    pts_list = list(flight_data.keys())
                    div_rows = []
                    for i in range(len(pts_list)):
                        for j in range(i + 1, len(pts_list)):
                            a, b = flight_data[pts_list[i]], flight_data[pts_list[j]]
                            # Compare at multiple checkpoints
                            for check_name, frac in [("Release", 0.0), ("1/3 Way", 0.33),
                                                      ("Commit Point", 0.6), ("2/3 Way", 0.67), ("Plate", 1.0)]:
                                idx_a = min(int(frac * (len(a["x"]) - 1)), len(a["x"]) - 1)
                                idx_b = min(int(frac * (len(b["x"]) - 1)), len(b["x"]) - 1)
                                h_sep = abs(a["y"][idx_a] - b["y"][idx_b]) * 12  # inches
                                s_sep = abs(a["x"][idx_a] - b["x"][idx_b]) * 12  # inches
                                total_sep = np.sqrt(h_sep**2 + s_sep**2)
                                div_rows.append({
                                    "Pair": f'{pts_list[i]} vs {pts_list[j]}',
                                    "Checkpoint": check_name,
                                    "Vertical Sep (in)": round(h_sep, 1),
                                    "Horizontal Sep (in)": round(s_sep, 1),
                                    "Total Sep (in)": round(total_sep, 1),
                                })

                    div_df = pd.DataFrame(div_rows)
                    # Show as line chart — separation over distance
                    fig_div = go.Figure()
                    for pair in div_df["Pair"].unique():
                        pair_data = div_df[div_df["Pair"] == pair]
                        checkpoints = ["Release", "1/3 Way", "Commit Point", "2/3 Way", "Plate"]
                        pair_ordered = pair_data.set_index("Checkpoint").reindex(checkpoints)
                        fig_div.add_trace(go.Scatter(
                            x=checkpoints,
                            y=pair_ordered["Total Sep (in)"].values,
                            mode="lines+markers",
                            name=pair,
                            line=dict(width=3),
                            marker=dict(size=8),
                        ))
                    fig_div.add_hline(y=6, line_dash="dash", line_color="#cc0000",
                                     annotation_text="6 in (hard to distinguish)",
                                     annotation_position="top left",
                                     annotation_font=dict(color="#cc0000"))
                    fig_div.update_layout(
                        height=380,
                        yaxis_title="Separation (inches)",
                        xaxis_title="Pitch Flight Checkpoint",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        **CHART_LAYOUT,
                    )
                    st.plotly_chart(fig_div, use_container_width=True)

    # ═══════════════════════════════════════════
    # TAB 6: COMMAND+ SCORING
    # ═══════════════════════════════════════════
    with tab_cmd:
        section_header("Command+ Analysis")
        st.caption("Command+ measures how well a pitcher locates each pitch relative to optimal zones. "
                   "100 = average command, higher = more precise location control.")

        # Compute Command+ using shared helper
        cmd_df = _compute_command_plus(pdf, data)

        if cmd_df.empty:
            st.info("Not enough location data to compute Command+.")
        else:
            # Display table
            st.dataframe(cmd_df, use_container_width=True, hide_index=True)

            # Command+ percentile bars
            section_header("Command+ Percentile Rankings")
            metrics = []
            for _, row in cmd_df.iterrows():
                cmd_val = row["Command+"]
                # Map Command+ to percentile-like scale for the bar
                pctl_mapped = min(max((cmd_val - 80) * 2.5, 0), 100)
                metrics.append((row["Pitch"], cmd_val, pctl_mapped, ".0f", True))
            if metrics:
                render_savant_percentile_section(metrics)

            # Location scatter per pitch type with density ellipse
            section_header("Location Precision Map")
            st.caption("Each dot is a pitch. Tight clusters = good command. The crosshair shows the average location.")
            loc_pitch_sel = st.selectbox("Select Pitch", cmd_df["Pitch"].tolist(), key="pdl_cmd_pitch")
            loc_ptd = pdf[(pdf["TaggedPitchType"] == loc_pitch_sel)].dropna(
                subset=["PlateLocSide", "PlateLocHeight"])

            if len(loc_ptd) >= 5:
                # Color by outcome
                outcome_colors = {
                    "StrikeSwinging": "#be0000",
                    "StrikeCalled": "#2d7fc1",
                    "BallCalled": "#9e9e9e",
                    "FoulBall": "#ee7e1e",
                    "InPlay": "#1dbe3a",
                    "FoulBallNotFieldable": "#ee7e1e",
                    "FoulBallFieldable": "#ee7e1e",
                    "HitByPitch": "#333",
                    "BallIntentional": "#9e9e9e",
                    "BallinDirt": "#666",
                }
                loc_ptd = loc_ptd.copy()
                loc_ptd["Outcome"] = loc_ptd["PitchCall"].map(
                    lambda x: "Whiff" if x == "StrikeSwinging" else
                              "Called Strike" if x == "StrikeCalled" else
                              "Ball" if "Ball" in str(x) else
                              "Foul" if "Foul" in str(x) else
                              "In Play" if x == "InPlay" else "Other"
                )
                outcome_color_map = {
                    "Whiff": "#be0000", "Called Strike": "#2d7fc1",
                    "Ball": "#9e9e9e", "Foul": "#ee7e1e",
                    "In Play": "#1dbe3a", "Other": "#666",
                }
                fig_loc = px.scatter(
                    loc_ptd, x="PlateLocSide", y="PlateLocHeight",
                    color="Outcome", color_discrete_map=outcome_color_map,
                    opacity=0.6,
                )
                # Mean crosshair
                mean_s = loc_ptd["PlateLocSide"].mean()
                mean_h = loc_ptd["PlateLocHeight"].mean()
                fig_loc.add_trace(go.Scatter(
                    x=[mean_s], y=[mean_h], mode="markers", showlegend=False,
                    marker=dict(size=20, color=PITCH_COLORS.get(loc_pitch_sel, "#333"),
                                symbol="x-thin", line=dict(width=4)),
                ))
                # 1-sigma ellipse
                std_s = loc_ptd["PlateLocSide"].std()
                std_h = loc_ptd["PlateLocHeight"].std()
                theta = np.linspace(0, 2 * np.pi, 50)
                fig_loc.add_trace(go.Scatter(
                    x=mean_s + std_s * np.cos(theta),
                    y=mean_h + std_h * np.sin(theta),
                    mode="lines", showlegend=False,
                    line=dict(color=PITCH_COLORS.get(loc_pitch_sel, "#333"),
                              width=2, dash="dash"),
                ))
                add_strike_zone(fig_loc)
                fig_loc.update_layout(
                    height=480,
                    xaxis=dict(range=[-2.5, 2.5], title="Plate Side (ft)", scaleanchor="y"),
                    yaxis=dict(range=[0, 5.5], title="Plate Height (ft)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    **CHART_LAYOUT,
                )
                st.plotly_chart(fig_loc, use_container_width=True)


    # ═══════════════════════════════════════════


# ──────────────────────────────────────────────
# HITTERS LAB HELPERS
# ──────────────────────────────────────────────

def _create_zone_grid_data(df, metric="swing_rate", batter_side="Right"):
    """Create 5x5 zone grid data for heatmaps.

    For RHH: negative PlateLocSide = inside, positive = outside.
    For LHH: negative PlateLocSide = outside, positive = inside.
    Returns (grid, annot, h_labels, v_labels) oriented from batter's perspective
    so column 0 = Inside for the given batter side.
    """
    h_edges = [-2, -0.83, -0.28, 0.28, 0.83, 2]
    v_edges = [0.5, 1.5, 2.17, 2.83, 3.5, 4.5]
    grid = np.full((5, 5), np.nan)
    annot = [['' for _ in range(5)] for _ in range(5)]
    for i in range(5):
        for j in range(5):
            zone_df = df[
                (df["PlateLocSide"] >= h_edges[i]) & (df["PlateLocSide"] < h_edges[i + 1]) &
                (df["PlateLocHeight"] >= v_edges[j]) & (df["PlateLocHeight"] < v_edges[j + 1])
            ]
            if len(zone_df) < 3:
                continue
            swings = zone_df[zone_df["PitchCall"].isin(SWING_CALLS)]
            whiffs = zone_df[zone_df["PitchCall"] == "StrikeSwinging"]
            contacts = zone_df[zone_df["PitchCall"].isin(CONTACT_CALLS)]
            batted = zone_df[zone_df["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            val = np.nan
            if metric == "swing_rate":
                val = len(swings) / len(zone_df) * 100
                annot[j][i] = f"{val:.0f}%"
            elif metric == "whiff_rate":
                if len(swings) > 0:
                    val = len(whiffs) / len(swings) * 100
                    annot[j][i] = f"{val:.0f}%"
            elif metric == "contact_rate":
                if len(swings) > 0:
                    val = len(contacts) / len(swings) * 100
                    annot[j][i] = f"{val:.0f}%"
            elif metric == "avg_ev":
                if len(batted) > 0:
                    val = batted["ExitSpeed"].mean()
                    annot[j][i] = f"{val:.0f}"
            grid[j, i] = val
    # For RHH: negative PlateLocSide = inside, so col 0 = Far In
    # For LHH: negative PlateLocSide = outside, so col 0 = Far Out
    # Keep data in physical plate coordinates; flip labels to match batter perspective
    if batter_side == "Left":
        h_labels = ["Far Out", "Outside", "Middle", "Inside", "Far In"]
    else:
        h_labels = ["Far In", "Inside", "Middle", "Outside", "Far Out"]
    v_labels = ["Low+", "Low", "Mid", "High", "High+"]
    return grid, annot, h_labels, v_labels


def _compute_expected_outcomes(batted_df):
    """Compute expected outcomes based on EV/LA buckets.

    Probabilities calibrated for D1 college baseball (lower HR rates,
    more singles than MLB).  wOBA weights use 2024 NCAA run-environment
    scaling (slightly lower than MLB linear weights).
    """
    if batted_df.empty:
        return {}
    outcomes = []
    for _, row in batted_df.iterrows():
        ev, la = row.get("ExitSpeed", 0), row.get("Angle", 0)
        if pd.isna(ev) or pd.isna(la):
            continue
        # Barrel zone — college HR rate ~30% vs MLB ~40%
        if is_barrel(ev, la):
            outcomes.append({"xOut": 0.30, "x1B": 0.10, "x2B": 0.22, "x3B": 0.05, "xHR": 0.33})
        # High-EV fly ball
        elif ev >= 95 and 25 <= la <= 45:
            outcomes.append({"xOut": 0.45, "x1B": 0.05, "x2B": 0.18, "x3B": 0.05, "xHR": 0.27})
        # Line drive / hard-hit
        elif 10 <= la <= 25 and ev >= 85:
            outcomes.append({"xOut": 0.28, "x1B": 0.47, "x2B": 0.20, "x3B": 0.03, "xHR": 0.02})
        # Ground ball
        elif la < 10:
            outcomes.append({"xOut": 0.76, "x1B": 0.22, "x2B": 0.02, "x3B": 0.00, "xHR": 0.00})
        # Pop-up / extreme fly ball
        elif la > 45:
            outcomes.append({"xOut": 0.95, "x1B": 0.03, "x2B": 0.01, "x3B": 0.00, "xHR": 0.01})
        # Soft contact
        elif ev < 70:
            outcomes.append({"xOut": 0.90, "x1B": 0.08, "x2B": 0.02, "x3B": 0.00, "xHR": 0.00})
        # Medium contact catchall
        else:
            outcomes.append({"xOut": 0.68, "x1B": 0.22, "x2B": 0.07, "x3B": 0.01, "xHR": 0.02})
    if not outcomes:
        return {}
    odf = pd.DataFrame(outcomes)
    # NCAA-adjusted wOBA weights (lower run environment than MLB)
    odf["xwOBAcon"] = (0.0 * odf["xOut"] + 0.88 * odf["x1B"] + 1.24 * odf["x2B"]
                       + 1.56 * odf["x3B"] + 2.0 * odf["xHR"])
    return odf.mean().to_dict()




# ──────────────────────────────────────────────
# HITTING LAB TAB CONTENT (used by page_hitting)
# ──────────────────────────────────────────────
def _hitting_lab_content(data, batter, season_filter, bdf, batted, pr, all_batter_stats,
                         tab_quality, tab_discipline, tab_coverage, tab_approach,
                         tab_pitch_type, tab_spray, tab_swing):
    """Render the 7 Hitters Lab tabs. Called from page_hitting()."""
    side = safe_mode(bdf["BatterSide"], "")
    _iz_mask = in_zone_mask(bdf)
    out_zone_mask = ~_iz_mask & bdf["PlateLocSide"].notna() & bdf["PlateLocHeight"].notna()

    # ─── Tab 1: Batted Ball Quality ─────────────────────
    with tab_quality:
        section_header("Batted Ball Quality Grades")
        if len(batted) < 3:
            st.info("Not enough batted ball data (need 3+ BBE).")
        else:
            col_pct, col_scatter = st.columns([1, 2])
            with col_pct:
                metrics_data = []
                for label, col_name, fmt, hib in [
                    ("Avg EV", "AvgEV", ".1f", True), ("Max EV", "MaxEV", ".1f", True),
                    ("Barrel%", "BarrelPct", ".1f", True), ("Hard-Hit%", "HardHitPct", ".1f", True),
                    ("Sweet Spot%", "SweetSpotPct", ".1f", True), ("Avg LA", "AvgLA", ".1f", True),
                    ("K%", "KPct", ".1f", False), ("BB%", "BBPct", ".1f", True),
                ]:
                    val = pr.get(col_name, np.nan)
                    pct = get_percentile(val, all_batter_stats[col_name].dropna()) if not pd.isna(val) else np.nan
                    metrics_data.append((label, val, pct, fmt, hib))
                render_savant_percentile_section(metrics_data, title="Percentile Rankings vs All Hitters")

            with col_scatter:
                section_header("Exit Velocity vs Launch Angle")
                bp = batted.copy()
                conditions = [
                    is_barrel_mask(bp),
                    bp["ExitSpeed"] >= 95,
                    bp["ExitSpeed"].between(80, 95),
                ]
                bp["Quality"] = np.select(conditions, ["Barrel", "Hard-Hit", "Medium"], default="Weak")
                q_colors = {"Barrel": "#d22d49", "Hard-Hit": "#fe6100", "Medium": "#f7c631", "Weak": "#aaaaaa"}
                fig_ev = px.scatter(bp, x="Angle", y="ExitSpeed", color="Quality",
                                    color_discrete_map=q_colors,
                                    labels={"Angle": "Launch Angle", "ExitSpeed": "Exit Velocity (mph)"})
                _bz_ev2 = np.linspace(98, max(batted["ExitSpeed"].max() + 2, 105), 40)
                _bz_la_lo2 = np.clip(26 - 2 * (_bz_ev2 - 98), 8, 26)
                _bz_la_hi2 = np.clip(30 + 3 * (_bz_ev2 - 98), 30, 50)
                fig_ev.add_trace(go.Scatter(
                    x=np.concatenate([_bz_la_lo2, _bz_la_hi2[::-1]]),
                    y=np.concatenate([_bz_ev2, _bz_ev2[::-1]]),
                    fill="toself", fillcolor="rgba(210,45,73,0.08)",
                    line=dict(color="rgba(210,45,73,0.3)", width=1, dash="dash"),
                    showlegend=False, hoverinfo="skip",
                ))
                fig_ev.add_annotation(x=20, y=_bz_ev2[-1], text="Barrel Zone",
                                       font=dict(size=9, color="#d22d49"), showarrow=False)
                fig_ev.update_layout(**CHART_LAYOUT, height=400,
                                      legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font=dict(size=10)))
                st.plotly_chart(fig_ev, use_container_width=True)

            col_ev_dist, col_la_dist = st.columns(2)
            with col_ev_dist:
                section_header("Exit Velocity Distribution")
                all_batted = query_population("SELECT ExitSpeed FROM trackman WHERE PitchCall='InPlay' AND ExitSpeed IS NOT NULL")
                fig_ev_hist = go.Figure()
                fig_ev_hist.add_trace(go.Histogram(
                    x=all_batted["ExitSpeed"], name="All Hitters",
                    marker_color="rgba(158,158,158,0.45)", nbinsx=30,
                    histnorm="probability density",
                ))
                fig_ev_hist.add_trace(go.Histogram(
                    x=batted["ExitSpeed"], name=display_name(batter),
                    marker_color="rgba(210,45,73,0.55)", nbinsx=25,
                    histnorm="probability density",
                ))
                fig_ev_hist.update_layout(
                    **CHART_LAYOUT, height=320, barmode="overlay",
                    xaxis_title="Exit Velocity (mph)", yaxis_title="Density",
                    legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font=dict(size=10)),
                )
                st.plotly_chart(fig_ev_hist, use_container_width=True)

            with col_la_dist:
                section_header("Launch Angle Distribution")
                la_bins = [(-90, -10, "Topped", "#d62728"), (-10, 8, "Ground Ball", "#ff7f0e"),
                           (8, 32, "Sweet Spot", "#2ca02c"), (32, 50, "Fly Ball", "#1f77b4"), (50, 90, "Popup", "#9467bd")]
                fig_la = go.Figure()
                for lo, hi, lbl, clr in la_bins:
                    subset = batted[batted["Angle"].between(lo, hi)]
                    fig_la.add_trace(go.Bar(x=[lbl], y=[len(subset)], name=lbl, marker_color=clr,
                                            text=[f"{len(subset)/len(batted)*100:.0f}%"], textposition="outside"))
                fig_la.update_layout(**CHART_LAYOUT, height=320, showlegend=False,
                                      yaxis_title="Count", xaxis_title="Launch Angle Zone", bargap=0.15)
                st.plotly_chart(fig_la, use_container_width=True)

            section_header("Expected Outcomes (EV/LA Model)")
            xo = _compute_expected_outcomes(batted)
            if xo:
                xo_cols = st.columns(6)
                for i, (k, lbl, clr) in enumerate([
                    ("xOut", "xOut%", "#9e9e9e"), ("x1B", "x1B%", "#2ca02c"), ("x2B", "x2B%", "#1f77b4"),
                    ("x3B", "x3B%", "#ff7f0e"), ("xHR", "xHR%", "#d22d49"), ("xwOBAcon", "xwOBAcon", "#6a0dad"),
                ]):
                    with xo_cols[i]:
                        val = xo.get(k, 0)
                        fmt_val = f"{val*100:.1f}%" if k != "xwOBAcon" else f"{val:.3f}"
                        st.markdown(
                            f'<div style="text-align:center;padding:12px;background:white;border-radius:8px;'
                            f'border:1px solid #eee;">'
                            f'<div style="font-size:24px;font-weight:800;color:{clr} !important;">{fmt_val}</div>'
                            f'<div style="font-size:11px;font-weight:600;color:#666 !important;text-transform:uppercase;">'
                            f'{lbl}</div></div>', unsafe_allow_html=True)

    # ─── Tab 2: Plate Discipline ────────────────────────
    with tab_discipline:
        section_header("Plate Discipline Overview")
        disc_metrics = []
        for label, col_name, fmt, hib in [
            ("Chase%", "ChasePct", ".1f", False), ("Whiff%", "WhiffPct", ".1f", False),
            ("K%", "KPct", ".1f", False), ("BB%", "BBPct", ".1f", True),
            ("Swing%", "SwingPct", ".1f", True), ("Z-Swing%", "ZoneSwingPct", ".1f", True),
            ("Z-Contact%", "ZoneContactPct", ".1f", True),
        ]:
            val = pr.get(col_name, np.nan)
            pct = get_percentile(val, all_batter_stats[col_name].dropna()) if not pd.isna(val) else np.nan
            disc_metrics.append((label, val, pct, fmt, hib))

        col_disc_pct, col_disc_grid = st.columns([1, 2])
        with col_disc_pct:
            render_savant_percentile_section(disc_metrics, title="Discipline Percentiles")
        with col_disc_grid:
            section_header("Swing Rate by Zone")
            grid_swing, annot_swing, h_labels, v_labels = _create_zone_grid_data(bdf, metric="swing_rate", batter_side=side)
            fig_grid = go.Figure(data=go.Heatmap(
                z=grid_swing, text=annot_swing, texttemplate="%{text}",
                x=h_labels, y=v_labels,
                colorscale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                zmin=0, zmax=100, showscale=True,
                colorbar=dict(title="Swing%", len=0.8),
            ))
            _add_grid_zone_outline(fig_grid)
            fig_grid.update_layout(**CHART_LAYOUT, height=380, xaxis=dict(side="bottom"))
            st.plotly_chart(fig_grid, use_container_width=True)

        col_ev_grid, col_chase = st.columns(2)
        with col_ev_grid:
            section_header("Avg EV by Zone")
            grid_ev, annot_ev, h_labels_ev, v_labels_ev = _create_zone_grid_data(bdf, metric="avg_ev", batter_side=side)
            fig_ev_grid = go.Figure(data=go.Heatmap(
                z=grid_ev, text=annot_ev, texttemplate="%{text}",
                x=h_labels_ev, y=v_labels_ev,
                colorscale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                zmin=60, zmax=100, showscale=True,
                colorbar=dict(title="EV", len=0.8),
            ))
            _add_grid_zone_outline(fig_ev_grid)
            fig_ev_grid.update_layout(**CHART_LAYOUT, height=380, xaxis=dict(side="bottom"))
            st.plotly_chart(fig_ev_grid, use_container_width=True)

        with col_chase:
            section_header("Chase Locations")
            chase_df = bdf[out_zone_mask].copy()
            if not chase_df.empty:
                chase_df["Outcome"] = "Taken"
                chase_df.loc[chase_df["PitchCall"] == "StrikeSwinging", "Outcome"] = "Swing & Miss"
                chase_df.loc[chase_df["PitchCall"].isin(["FoulBall", "FoulBallNotFieldable", "FoulBallFieldable"]), "Outcome"] = "Foul"
                chase_df.loc[chase_df["PitchCall"] == "InPlay", "Outcome"] = "In Play"
                chase_colors = {"Taken": "#aaaaaa", "Swing & Miss": "#d22d49", "Foul": "#f7c631", "In Play": "#2ca02c"}
                fig_chase = px.scatter(chase_df.dropna(subset=["PlateLocSide", "PlateLocHeight"]),
                                        x="PlateLocSide", y="PlateLocHeight", color="Outcome",
                                        color_discrete_map=chase_colors, opacity=0.6,
                                        labels={"PlateLocSide": "Horizontal", "PlateLocHeight": "Vertical"})
                add_strike_zone(fig_chase)
                fig_chase.update_layout(**CHART_LAYOUT, height=380,
                                         xaxis=dict(range=[-2.5, 2.5], scaleanchor="y"), yaxis=dict(range=[0, 5]),
                                         legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font=dict(size=10)))
                st.plotly_chart(fig_chase, use_container_width=True)
            else:
                st.info("No out-of-zone data available.")

        section_header("Pitch Recognition by Type")
        pt_disc_rows = []
        for pt in sorted(bdf["TaggedPitchType"].dropna().unique()):
            pt_df = bdf[bdf["TaggedPitchType"] == pt]
            if len(pt_df) < 5:
                continue
            pt_sw = pt_df[pt_df["PitchCall"].isin(SWING_CALLS)]
            pt_wh = pt_df[pt_df["PitchCall"] == "StrikeSwinging"]
            pt_ct = pt_df[pt_df["PitchCall"].isin(CONTACT_CALLS)]
            pt_oz = pt_df[(~in_zone_mask(pt_df)) & pt_df["PlateLocSide"].notna()]
            pt_ch = pt_oz[pt_oz["PitchCall"].isin(SWING_CALLS)]
            pt_disc_rows.append({
                "Pitch Type": pt, "Seen": len(pt_df),
                "Swing%": f"{len(pt_sw)/len(pt_df)*100:.1f}",
                "Whiff%": f"{len(pt_wh)/max(len(pt_sw),1)*100:.1f}" if len(pt_sw) > 0 else "-",
                "Contact%": f"{len(pt_ct)/max(len(pt_sw),1)*100:.1f}" if len(pt_sw) > 0 else "-",
                "Chase%": f"{len(pt_ch)/max(len(pt_oz),1)*100:.1f}" if len(pt_oz) > 0 else "-",
            })
        if pt_disc_rows:
            st.dataframe(pd.DataFrame(pt_disc_rows).set_index("Pitch Type"), use_container_width=True)

    # ─── Tab 3: Zone Coverage ───────────────────────────
    with tab_coverage:
        section_header("Zone Coverage Analysis")
        if len(batted) < 5:
            st.info("Not enough batted ball data for zone coverage analysis.")
        else:
            col_contact_hz, col_damage_hz = st.columns(2)
            with col_contact_hz:
                section_header("Contact Rate Heatmap")
                contact_loc = bdf[bdf["PitchCall"].isin(CONTACT_CALLS)].dropna(subset=["PlateLocSide", "PlateLocHeight"])
                if not contact_loc.empty:
                    fig_contact = go.Figure()
                    fig_contact.add_trace(go.Histogram2dContour(
                        x=contact_loc["PlateLocSide"], y=contact_loc["PlateLocHeight"],
                        colorscale=[[0, "rgba(255,255,255,0)"], [0.3, "#a8d5e2"], [0.6, "#f7c631"], [1, "#d22d49"]],
                        contours=dict(showlines=False), ncontours=15, showscale=False))
                    fig_contact.add_trace(go.Scatter(
                        x=contact_loc["PlateLocSide"], y=contact_loc["PlateLocHeight"],
                        mode="markers", marker=dict(size=4, color="#2ca02c", opacity=0.3),
                        showlegend=False, hoverinfo="skip"))
                    add_strike_zone(fig_contact)
                    fig_contact.update_layout(**CHART_LAYOUT, height=400,
                                               xaxis=dict(range=[-2.5, 2.5], title="Horizontal", scaleanchor="y"),
                                               yaxis=dict(range=[0, 5], title="Vertical"))
                    st.plotly_chart(fig_contact, use_container_width=True)

            with col_damage_hz:
                section_header("Damage Heatmap (Avg EV)")
                batted_loc = batted.dropna(subset=["PlateLocSide", "PlateLocHeight"])
                if not batted_loc.empty:
                    fig_damage = go.Figure()
                    fig_damage.add_trace(go.Histogram2dContour(
                        x=batted_loc["PlateLocSide"], y=batted_loc["PlateLocHeight"],
                        z=batted_loc["ExitSpeed"], histfunc="avg",
                        colorscale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                        contours=dict(showlines=False), ncontours=12, showscale=True,
                        colorbar=dict(title="Avg EV", len=0.8)))
                    barrel_loc = batted_loc[is_barrel_mask(batted_loc)]
                    if not barrel_loc.empty:
                        fig_damage.add_trace(go.Scatter(
                            x=barrel_loc["PlateLocSide"], y=barrel_loc["PlateLocHeight"],
                            mode="markers", marker=dict(size=12, color="#d22d49", symbol="star",
                                                         line=dict(width=1, color="white")),
                            name="Barrels", hovertemplate="EV: %{customdata[0]:.1f}<extra></extra>",
                            customdata=barrel_loc[["ExitSpeed"]].values))
                    add_strike_zone(fig_damage)
                    fig_damage.update_layout(**CHART_LAYOUT, height=400,
                                              xaxis=dict(range=[-2.5, 2.5], title="Horizontal", scaleanchor="y"),
                                              yaxis=dict(range=[0, 5], title="Vertical"),
                                              legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font=dict(size=10)))
                    st.plotly_chart(fig_damage, use_container_width=True)

            col_whiff_hz, _ = st.columns(2)
            with col_whiff_hz:
                section_header("Whiff Zone Map")
                grid_whiff, annot_whiff, h_lbl, v_lbl = _create_zone_grid_data(bdf, metric="whiff_rate", batter_side=side)
                fig_wz = go.Figure(data=go.Heatmap(
                    z=grid_whiff, text=annot_whiff, texttemplate="%{text}",
                    x=h_lbl, y=v_lbl,
                    colorscale=[[0, "#2ca02c"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                    zmin=0, zmax=60, showscale=True,
                    colorbar=dict(title="Whiff%", len=0.8)))
                _add_grid_zone_outline(fig_wz)
                fig_wz.update_layout(**CHART_LAYOUT, height=380)
                st.plotly_chart(fig_wz, use_container_width=True)

            section_header("Damage by Pitch Type & Location")
            top_pts = [pt for pt in sorted(bdf["TaggedPitchType"].dropna().unique())
                       if len(bdf[bdf["TaggedPitchType"] == pt]) >= 10][:4]
            if top_pts:
                pt_cols = st.columns(len(top_pts))
                for idx, pt in enumerate(top_pts):
                    with pt_cols[idx]:
                        pt_b = bdf[(bdf["TaggedPitchType"] == pt) & (bdf["PitchCall"] == "InPlay")].dropna(
                            subset=["ExitSpeed", "PlateLocSide", "PlateLocHeight"])
                        st.markdown(f'<div style="text-align:center;font-weight:700;font-size:13px;color:#1a1a2e !important;'
                                    f'margin-bottom:4px;">{pt} ({len(pt_b)} BBE)</div>', unsafe_allow_html=True)
                        if len(pt_b) >= 3:
                            fig_pt = go.Figure()
                            fig_pt.add_trace(go.Scatter(
                                x=pt_b["PlateLocSide"], y=pt_b["PlateLocHeight"], mode="markers",
                                marker=dict(size=8, color=pt_b["ExitSpeed"],
                                            colorscale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                                            cmin=60, cmax=100,
                                            showscale=(idx == len(top_pts) - 1),
                                            colorbar=dict(title="EV", len=0.8) if idx == len(top_pts) - 1 else None,
                                            line=dict(width=0.5, color="white")),
                                hovertemplate="EV: %{marker.color:.1f}<extra></extra>"))
                            add_strike_zone(fig_pt)
                            fig_pt.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                                  font=dict(size=11, color="#1a1a2e", family="Inter, Arial, sans-serif"),
                                                  height=300,
                                                  xaxis=dict(range=[-2.5, 2.5], showticklabels=False),
                                                  yaxis=dict(range=[0, 5], showticklabels=False),
                                                  margin=dict(l=5, r=5, t=5, b=5))
                            st.plotly_chart(fig_pt, use_container_width=True)
                        else:
                            st.caption("Not enough data")

    # ─── Tab 4: Approach Analysis ───────────────────────
    with tab_approach:
        section_header("Count-Based Approach")
        bdf_c = bdf.dropna(subset=["Balls", "Strikes"]).copy()
        bdf_c["Balls"] = bdf_c["Balls"].astype(int)
        bdf_c["Strikes"] = bdf_c["Strikes"].astype(int)

        count_grid_ev = np.full((4, 3), np.nan)
        count_grid_sw = np.full((4, 3), np.nan)
        annot_ev = [['' for _ in range(3)] for _ in range(4)]
        annot_sw = [['' for _ in range(3)] for _ in range(4)]
        for b in range(4):
            for s in range(3):
                cd = bdf_c[(bdf_c["Balls"] == b) & (bdf_c["Strikes"] == s)]
                if len(cd) < 3:
                    continue
                cb = cd[cd["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
                cs = cd[cd["PitchCall"].isin(SWING_CALLS)]
                if len(cb) > 0:
                    ev_v = cb["ExitSpeed"].mean()
                    count_grid_ev[b, s] = ev_v
                    annot_ev[b][s] = f"{ev_v:.0f}"
                sw_v = len(cs) / len(cd) * 100
                count_grid_sw[b, s] = sw_v
                annot_sw[b][s] = f"{sw_v:.0f}%"

        col_evc, col_swc = st.columns(2)
        with col_evc:
            section_header("Avg EV by Count")
            fig_evc = go.Figure(data=go.Heatmap(
                z=count_grid_ev, text=annot_ev, texttemplate="%{text}",
                x=["0 Strikes", "1 Strike", "2 Strikes"], y=["0 Balls", "1 Ball", "2 Balls", "3 Balls"],
                colorscale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                zmin=70, zmax=100, showscale=True, colorbar=dict(title="EV", len=0.8)))
            fig_evc.update_layout(**CHART_LAYOUT, height=320)
            st.plotly_chart(fig_evc, use_container_width=True)
        with col_swc:
            section_header("Swing% by Count")
            fig_swc = go.Figure(data=go.Heatmap(
                z=count_grid_sw, text=annot_sw, texttemplate="%{text}",
                x=["0 Strikes", "1 Strike", "2 Strikes"], y=["0 Balls", "1 Ball", "2 Balls", "3 Balls"],
                colorscale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                zmin=0, zmax=100, showscale=True, colorbar=dict(title="Swing%", len=0.8)))
            fig_swc.update_layout(**CHART_LAYOUT, height=320)
            st.plotly_chart(fig_swc, use_container_width=True)

        col_fp, col_2k = st.columns(2)
        with col_fp:
            section_header("First Pitch Performance")
            fp = bdf[bdf["PitchofPA"] == 1] if "PitchofPA" in bdf.columns else pd.DataFrame()
            if not fp.empty and len(fp) >= 5:
                fp_sw = fp[fp["PitchCall"].isin(SWING_CALLS)]
                fp_wh = fp[fp["PitchCall"] == "StrikeSwinging"]
                fp_bt = fp[fp["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
                for lbl, val in [("1st Pitch Swing%", len(fp_sw)/len(fp)*100),
                                  ("1st Pitch Whiff%", len(fp_wh)/max(len(fp_sw),1)*100 if len(fp_sw) > 0 else 0),
                                  ("1st Pitch Avg EV", fp_bt["ExitSpeed"].mean() if len(fp_bt) > 0 else np.nan),
                                  ("1st Pitch BBE", float(len(fp_bt)))]:
                    fv = f"{val:.1f}" if not pd.isna(val) else "-"
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;padding:8px 12px;'
                        f'background:white;border-radius:6px;margin:4px 0;border:1px solid #eee;">'
                        f'<span style="font-size:12px;font-weight:600;color:#1a1a2e !important;">{lbl}</span>'
                        f'<span style="font-size:14px;font-weight:800;color:#d22d49 !important;">{fv}</span></div>',
                        unsafe_allow_html=True)
            else:
                st.info("Not enough first-pitch data.")

        with col_2k:
            section_header("2-Strike Adjustments")
            early = bdf_c[bdf_c["Strikes"] < 2]
            two_k = bdf_c[bdf_c["Strikes"] == 2]
            if len(early) >= 10 and len(two_k) >= 10:
                rows_2k = []
                for lbl, fn in [
                    ("Swing%", lambda d: len(d[d["PitchCall"].isin(SWING_CALLS)])/max(len(d),1)*100),
                    ("Whiff%", lambda d: len(d[d["PitchCall"]=="StrikeSwinging"])/max(len(d[d["PitchCall"].isin(SWING_CALLS)]),1)*100 if len(d[d["PitchCall"].isin(SWING_CALLS)])>0 else 0),
                ]:
                    ev = fn(early)
                    tv = fn(two_k)
                    rows_2k.append({"Metric": lbl, "<2 Strikes": f"{ev:.1f}%", "2 Strikes": f"{tv:.1f}%", "Change": f"{tv-ev:+.1f}%"})
                st.dataframe(pd.DataFrame(rows_2k).set_index("Metric"), use_container_width=True)
            else:
                st.info("Not enough data for 2-strike analysis.")

        section_header("At-Bat Length Outcomes")
        if "PitchofPA" in bdf.columns:
            pa_id_cols = [c for c in ["GameID", "PAofInning", "Inning", "Batter"] if c in bdf.columns]
            if len(pa_id_cols) >= 2:
                pa_lens = bdf.groupby(pa_id_cols)["PitchofPA"].max()
                length_rows = []
                for lo, hi, lbl in [(1, 3, "1-3 pitches"), (4, 6, "4-6 pitches"), (7, 20, "7+ pitches")]:
                    pa_ids = pa_lens[(pa_lens >= lo) & (pa_lens <= hi)]
                    n_pa = len(pa_ids)
                    if n_pa == 0:
                        continue
                    pa_sub = bdf.set_index(pa_id_cols).loc[pa_ids.index].reset_index()
                    ks = pa_sub[pa_sub["KorBB"] == "Strikeout"].drop_duplicates(subset=pa_id_cols) if "KorBB" in pa_sub.columns else pd.DataFrame()
                    bbs = pa_sub[pa_sub["KorBB"] == "Walk"].drop_duplicates(subset=pa_id_cols) if "KorBB" in pa_sub.columns else pd.DataFrame()
                    bbe = pa_sub[pa_sub["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
                    length_rows.append({
                        "PA Length": lbl, "PA": n_pa,
                        "K%": f"{len(ks)/n_pa*100:.1f}%",
                        "BB%": f"{len(bbs)/n_pa*100:.1f}%",
                        "Avg EV": f"{bbe['ExitSpeed'].mean():.1f}" if len(bbe) > 0 else "-",
                    })
                if length_rows:
                    st.dataframe(pd.DataFrame(length_rows).set_index("PA Length"), use_container_width=True)

    # ─── Tab 5: Pitch Type Performance ──────────────────
    with tab_pitch_type:
        section_header("Performance by Pitch Type")
        pt_rows = []
        for pt in sorted(bdf["TaggedPitchType"].dropna().unique()):
            ptd = bdf[bdf["TaggedPitchType"] == pt]
            if len(ptd) < 5:
                continue
            pt_sw = ptd[ptd["PitchCall"].isin(SWING_CALLS)]
            pt_wh = ptd[ptd["PitchCall"] == "StrikeSwinging"]
            pt_ct = ptd[ptd["PitchCall"].isin(CONTACT_CALLS)]
            pt_bt = ptd[ptd["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            pt_br = pt_bt[is_barrel_mask(pt_bt)] if len(pt_bt) > 0 else pd.DataFrame()
            pt_rows.append({
                "Pitch Type": pt, "Seen": len(ptd),
                "Seen%": len(ptd)/len(bdf)*100,
                "Swing%": len(pt_sw)/len(ptd)*100,
                "Whiff%": len(pt_wh)/max(len(pt_sw),1)*100 if len(pt_sw) > 0 else 0,
                "Contact%": len(pt_ct)/max(len(pt_sw),1)*100 if len(pt_sw) > 0 else 0,
                "BBE": len(pt_bt),
                "Avg EV": pt_bt["ExitSpeed"].mean() if len(pt_bt) > 0 else np.nan,
                "Max EV": pt_bt["ExitSpeed"].max() if len(pt_bt) > 0 else np.nan,
                "Barrel%": len(pt_br)/max(len(pt_bt),1)*100 if len(pt_bt) > 0 else 0,
            })

        if pt_rows:
            pt_df = pd.DataFrame(pt_rows)
            disp = pt_df.copy()
            for c in ["Seen%", "Swing%", "Whiff%", "Contact%", "Barrel%"]:
                disp[c] = disp[c].map(lambda x: f"{x:.1f}%")
            for c in ["Avg EV", "Max EV"]:
                disp[c] = disp[c].map(lambda x: f"{x:.1f}" if not pd.isna(x) else "-")
            st.dataframe(disp.set_index("Pitch Type"), use_container_width=True)

            col_evb, col_whb = st.columns(2)
            with col_evb:
                section_header("Avg EV by Pitch Type")
                ch = pt_df.dropna(subset=["Avg EV"]).sort_values("Avg EV", ascending=True)
                if not ch.empty:
                    colors = [PITCH_COLORS.get(p, "#aaa") for p in ch["Pitch Type"]]
                    fig_pe = go.Figure(go.Bar(
                        y=ch["Pitch Type"], x=ch["Avg EV"], orientation="h", marker_color=colors,
                        text=ch["Avg EV"].map(lambda x: f"{x:.1f}"), textposition="outside"))
                    fig_pe.update_layout(**CHART_LAYOUT, height=max(200, len(ch)*40+60),
                                          xaxis_title="Avg Exit Velocity (mph)",
                                          xaxis=dict(range=[60, ch["Avg EV"].max()+8]))
                    st.plotly_chart(fig_pe, use_container_width=True)
            with col_whb:
                section_header("Whiff% by Pitch Type")
                ch2 = pt_df.sort_values("Whiff%", ascending=True)
                colors = [PITCH_COLORS.get(p, "#aaa") for p in ch2["Pitch Type"]]
                fig_pw = go.Figure(go.Bar(
                    y=ch2["Pitch Type"], x=ch2["Whiff%"], orientation="h", marker_color=colors,
                    text=ch2["Whiff%"].map(lambda x: f"{x:.1f}%"), textposition="outside"))
                fig_pw.update_layout(**CHART_LAYOUT, height=max(200, len(ch2)*40+60),
                                      xaxis_title="Whiff%", xaxis=dict(range=[0, max(ch2["Whiff%"].max()+10, 40)]))
                st.plotly_chart(fig_pw, use_container_width=True)

            section_header("Pitch Movement vs Damage")
            bwm = bdf[(bdf["PitchCall"] == "InPlay")].dropna(subset=["ExitSpeed", "HorzBreak", "InducedVertBreak"])
            if len(bwm) >= 5:
                fig_mv = px.scatter(bwm, x="HorzBreak", y="InducedVertBreak", color="ExitSpeed",
                                     size="ExitSpeed",
                                     color_continuous_scale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                                     hover_data={"TaggedPitchType": True, "ExitSpeed": ":.1f"},
                                     labels={"HorzBreak": "Horizontal Break (in)", "InducedVertBreak": "Induced Vert Break (in)", "ExitSpeed": "EV"})
                fig_mv.update_layout(**CHART_LAYOUT, height=400)
                st.plotly_chart(fig_mv, use_container_width=True)
        else:
            st.info("Not enough pitch type data.")

    # ─── Tab 6: Spray Lab ──────────────────────────────
    with tab_spray:
        section_header("Spray Chart Analysis")
        in_play = bdf[bdf["PitchCall"] == "InPlay"].copy()
        if len(in_play) < 5:
            st.info("Not enough in-play data for spray analysis.")
        else:
            col_spray, col_table = st.columns([2, 1])
            with col_spray:
                spray_data = in_play.dropna(subset=["Direction", "Distance"]).copy()
                if not spray_data.empty:
                    angle_rad = np.radians(spray_data["Direction"])
                    spray_data["x"] = spray_data["Distance"] * np.sin(angle_rad)
                    spray_data["y"] = spray_data["Distance"] * np.cos(angle_rad)
                    fig_sp = go.Figure()
                    theta_g = np.linspace(-np.pi/4, np.pi/4, 80)
                    gr = 400
                    fig_sp.add_trace(go.Scatter(x=[0]+list(gr*np.sin(theta_g))+[0],
                                                y=[0]+list(gr*np.cos(theta_g))+[0], mode="lines",
                                                fill="toself", fillcolor="rgba(76,160,60,0.06)",
                                                line=dict(color="rgba(76,160,60,0.15)", width=1),
                                                showlegend=False, hoverinfo="skip"))
                    fig_sp.add_trace(go.Scatter(x=[0,63.6,0,-63.6,0], y=[0,63.6,127.3,63.6,0], mode="lines",
                                                line=dict(color="rgba(160,120,60,0.25)", width=1),
                                                fill="toself", fillcolor="rgba(160,120,60,0.06)",
                                                showlegend=False, hoverinfo="skip"))
                    fl = 350
                    for sx in [-1, 1]:
                        fig_sp.add_trace(go.Scatter(x=[0, sx*fl*np.sin(np.pi/4)], y=[0, fl*np.cos(np.pi/4)],
                                                    mode="lines", line=dict(color="rgba(0,0,0,0.12)", width=1),
                                                    showlegend=False, hoverinfo="skip"))
                    ev_vals = spray_data["ExitSpeed"].fillna(0)
                    fig_sp.add_trace(go.Scatter(
                        x=spray_data["x"], y=spray_data["y"], mode="markers",
                        marker=dict(size=8, color=ev_vals,
                                    colorscale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                                    cmin=60, cmax=100, showscale=True, colorbar=dict(title="EV", len=0.6),
                                    line=dict(width=0.5, color="white")),
                        hovertemplate="EV: %{marker.color:.1f} mph<br>Dist: %{customdata[0]:.0f}ft<extra></extra>",
                        customdata=spray_data[["Distance"]].values, showlegend=False))
                    fig_sp.update_layout(
                        xaxis=dict(range=[-300, 300], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
                        yaxis=dict(range=[-15, 400], showgrid=False, zeroline=False, showticklabels=False,
                                   scaleanchor="x", fixedrange=True),
                        height=450, margin=dict(l=0, r=0, t=5, b=0),
                        plot_bgcolor="white", paper_bgcolor="white")
                    st.plotly_chart(fig_sp, use_container_width=True)

            with col_table:
                section_header("Pull / Center / Oppo")
                bd = in_play.dropna(subset=["Direction", "ExitSpeed"]).copy()
                if not bd.empty:
                    bs = safe_mode(bdf["BatterSide"], "Right")
                    if bs == "Left":
                        pull_m = bd["Direction"] > 15
                        oppo_m = bd["Direction"] < -15
                    else:
                        pull_m = bd["Direction"] < -15
                        oppo_m = bd["Direction"] > 15
                    center_m = ~pull_m & ~oppo_m
                    dir_rows = []
                    for lbl, mask in [("Pull", pull_m), ("Center", center_m), ("Oppo", oppo_m)]:
                        sub = bd[mask]
                        ns = len(sub)
                        if ns == 0:
                            dir_rows.append({"Dir": lbl, "BBE": 0, "%": "-", "Avg EV": "-", "Max EV": "-"})
                            continue
                        gb_n = len(sub[sub["TaggedHitType"] == "GroundBall"])
                        ld_n = len(sub[sub["TaggedHitType"] == "LineDrive"])
                        fb_n = len(sub[sub["TaggedHitType"] == "FlyBall"])
                        brr = int(is_barrel_mask(sub).sum()) if sub["Angle"].notna().any() else 0
                        dir_rows.append({
                            "Dir": lbl, "BBE": ns,
                            "%": f"{ns/len(bd)*100:.0f}%",
                            "Avg EV": f"{sub['ExitSpeed'].mean():.1f}",
                            "Max EV": f"{sub['ExitSpeed'].max():.1f}",
                            "GB%": f"{gb_n/ns*100:.0f}%",
                            "LD%": f"{ld_n/ns*100:.0f}%",
                            "FB%": f"{fb_n/ns*100:.0f}%",
                            "Barrel": brr,
                        })
                    st.dataframe(pd.DataFrame(dir_rows).set_index("Dir"), use_container_width=True)

            col_la_dir, col_gb = st.columns(2)
            with col_la_dir:
                section_header("Launch Angle by Direction")
                bla = in_play.dropna(subset=["Direction", "Angle"]).copy()
                if not bla.empty:
                    bs = safe_mode(bdf["BatterSide"], "Right")
                    if bs == "Left":
                        bla["Field"] = np.where(bla["Direction"] > 15, "Pull",
                                                 np.where(bla["Direction"] < -15, "Oppo", "Center"))
                    else:
                        bla["Field"] = np.where(bla["Direction"] < -15, "Pull",
                                                 np.where(bla["Direction"] > 15, "Oppo", "Center"))
                    fig_ld = px.box(bla, x="Field", y="Angle",
                                     category_orders={"Field": ["Pull", "Center", "Oppo"]},
                                     color="Field", color_discrete_map={"Pull": "#d22d49", "Center": "#9e9e9e", "Oppo": "#1f77b4"},
                                     labels={"Angle": "Launch Angle", "Field": ""})
                    fig_ld.add_shape(type="rect", x0=-0.5, x1=2.5, y0=8, y1=32,
                                      fillcolor="rgba(29,190,58,0.08)", line=dict(width=0))
                    fig_ld.add_annotation(x=2.3, y=20, text="Sweet Spot", font=dict(size=9, color="#2ca02c"), showarrow=False)
                    fig_ld.update_layout(**CHART_LAYOUT, height=350, showlegend=False)
                    st.plotly_chart(fig_ld, use_container_width=True)
            with col_gb:
                section_header("GB% by Pitch Type")
                gb_rows = []
                for pt in sorted(bdf["TaggedPitchType"].dropna().unique()):
                    pt_ip = in_play[in_play["TaggedPitchType"] == pt]
                    if len(pt_ip) < 3:
                        continue
                    gb_n = len(pt_ip[pt_ip["TaggedHitType"] == "GroundBall"])
                    gb_rows.append({"Pitch Type": pt, "GB%": gb_n/len(pt_ip)*100, "BBE": len(pt_ip)})
                if gb_rows:
                    gdf = pd.DataFrame(gb_rows).sort_values("GB%", ascending=True)
                    colors = [PITCH_COLORS.get(p, "#aaa") for p in gdf["Pitch Type"]]
                    fig_gb = go.Figure(go.Bar(
                        y=gdf["Pitch Type"], x=gdf["GB%"], orientation="h", marker_color=colors,
                        text=gdf["GB%"].map(lambda x: f"{x:.0f}%"), textposition="outside"))
                    fig_gb.update_layout(**CHART_LAYOUT, height=max(200, len(gdf)*40+60),
                                          xaxis_title="Ground Ball %", xaxis=dict(range=[0, 100]))
                    st.plotly_chart(fig_gb, use_container_width=True)

    # ─── Tab 7: Swing Path Analysis ─────────────────────
    with tab_swing:
        section_header("Swing Path Analysis")
        st.caption("Bat path reconstructed from hard-hit ball physics, pitch-height regression, bat speed estimation, and contact-depth analysis")

        swings = bdf[bdf["PitchCall"].isin(SWING_CALLS)].copy()
        whiffs = bdf[bdf["PitchCall"] == "StrikeSwinging"].copy()
        contacts = bdf[bdf["PitchCall"].isin(CONTACT_CALLS)].copy()
        inplay = bdf[(bdf["PitchCall"] == "InPlay")].copy()
        inplay_ev = inplay.dropna(subset=["ExitSpeed", "PlateLocSide", "PlateLocHeight"])
        _bs = safe_mode(bdf["BatterSide"], "Right") if "BatterSide" in bdf.columns else "Right"

        if len(swings) < 10:
            st.info("Not enough swing data (need 10+ swings).")
        else:
            from scipy import stats as sp_stats

            # ── Core data: all in-play with EV + LA + location ──
            inplay_full = inplay.dropna(subset=["Angle", "ExitSpeed", "PlateLocHeight", "PlateLocSide"]).copy()

            if len(inplay_full) >= 10:
                # ── Hard-hit selection: top 25% EV (best attack angle proxy) ──
                ev_75 = inplay_full["ExitSpeed"].quantile(0.75)
                hard_hit = inplay_full[inplay_full["ExitSpeed"] >= ev_75].copy()
                all_hit = inplay_full.copy()

                # ── VERTICAL ATTACK ANGLE ──
                # Primary: median LA on hard-hit = best proxy for bat path angle
                # On squared-up contact, ball exits roughly along bat's travel direction
                attack_angle = hard_hit["Angle"].median()
                avg_la_all = all_hit["Angle"].median()
                # EV²-weighted mean gives extra emphasis to absolute best contact
                hard_hit_w = np.average(hard_hit["Angle"], weights=hard_hit["ExitSpeed"]**2)

                # Regression: LA vs pitch height on hard-hit balls
                # Slope = path adjustment rate (how much hitter changes plane per ft of pitch height)
                # Value at 2.5ft = mid-zone baseline attack angle
                if len(hard_hit) >= 3:
                    v_slope, v_int, v_r, _, _ = sp_stats.linregress(hard_hit["PlateLocHeight"].values, hard_hit["Angle"].values)
                    mid_zone_aa = v_int + v_slope * 2.5
                    path_adjust = v_slope
                else:
                    v_slope, v_int, v_r = 0, attack_angle, 0
                    mid_zone_aa = attack_angle
                    path_adjust = 0

                # ── HORIZONTAL BAT PATH ──
                # Direction vs PlateLocSide on hard-hit: slope = horizontal path tendency
                h_data = hard_hit.dropna(subset=["Direction"])
                if len(h_data) >= 5:
                    h_slope, h_int, h_r, _, _ = sp_stats.linregress(h_data["PlateLocSide"].values, h_data["Direction"].values)
                    # Positive slope for RHH = pulls inside pitches more (natural)
                    # h_int at PlateLocSide=0 = center-field tendency
                    horz_center = h_int  # spray direction on center-plate pitch
                    horz_adjust = h_slope  # deg/ft of horizontal location
                    horz_r2 = h_r**2
                else:
                    horz_center, horz_adjust, horz_r2 = np.nan, np.nan, np.nan

                # ── BAT SPEED PROXY ──
                # Empirical model: EV ≈ 0.2 * pitch_speed + 1.2 * bat_speed
                if "RelSpeed" in hard_hit.columns and hard_hit["RelSpeed"].notna().any():
                    hard_hit["BatSpeedProxy"] = (hard_hit["ExitSpeed"] - 0.2 * hard_hit["RelSpeed"]) / 1.2
                    bat_speed_avg = hard_hit["BatSpeedProxy"].mean()
                    bat_speed_max = hard_hit["BatSpeedProxy"].max()
                    # DB comparison
                    all_db_inplay = query_population("SELECT ExitSpeed, RelSpeed FROM trackman WHERE PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND RelSpeed IS NOT NULL")
                    if len(all_db_inplay) > 0:
                        all_db_ev75 = all_db_inplay["ExitSpeed"].quantile(0.75)
                        all_db_hh = all_db_inplay[all_db_inplay["ExitSpeed"] >= all_db_ev75]
                        all_db_hh_bs = (all_db_hh["ExitSpeed"] - 0.2 * all_db_hh["RelSpeed"]) / 1.2
                        bat_speed_pctile = (all_db_hh_bs < bat_speed_avg).mean() * 100
                    else:
                        bat_speed_pctile = np.nan
                else:
                    bat_speed_avg = bat_speed_max = bat_speed_pctile = np.nan

                # ── CONTACT DEPTH PROXY ──
                # EffectiveVelo accounts for extension — higher EffVelo means earlier contact point
                # ContactDepth = EffectiveVelo - RelSpeed: more negative = deeper contact (closer to catcher)
                if "EffectiveVelo" in hard_hit.columns and hard_hit["EffectiveVelo"].notna().any():
                    hard_hit["ContactDepth"] = hard_hit["EffectiveVelo"] - hard_hit["RelSpeed"]
                    contact_depth = hard_hit["ContactDepth"].mean()
                    # Positive = out front, negative = deeper
                    if contact_depth > 0:
                        depth_label = "Out Front"
                        depth_desc = "Gets to the ball early — turns on inside pitches, may be early on off-speed"
                    elif contact_depth > -1.5:
                        depth_label = "Neutral"
                        depth_desc = "Average timing depth — balanced contact point"
                    else:
                        depth_label = "Deep Contact"
                        depth_desc = "Lets the ball travel — strong to opposite field, may struggle with inside velocity"
                else:
                    contact_depth = np.nan
                    depth_label = depth_desc = None

                # ── Swing type classification ──
                if attack_angle > 20:
                    swing_type, swing_color = "Steep Uppercut", "#d22d49"
                    swing_desc = "Extreme loft — high HR upside but vulnerable to high fastballs and off-speed below zone"
                elif attack_angle > 12:
                    swing_type, swing_color = "Lift-Oriented", "#fe6100"
                    swing_desc = "Modern power swing — generates carry and hard fly balls consistently"
                elif attack_angle > 5:
                    swing_type, swing_color = "Slight Uppercut", "#f7c631"
                    swing_desc = "Balanced loft — matches most pitch planes well, produces line drives and fly balls"
                elif attack_angle > -2:
                    swing_type, swing_color = "Level", "#2ca02c"
                    swing_desc = "Flat bat path — contact-oriented, line drive approach with gap-to-gap power"
                else:
                    swing_type, swing_color = "Downward / Chopper", "#1f77b4"
                    swing_desc = "Downward swing plane — ground ball heavy, limits hard fly ball contact"

                # ═══ SECTION 1: Attack Angle Overview ═══
                section_header("Vertical Attack Angle")
                st.caption("Median launch angle on hard-hit balls (EV ≥ 75th pctile). On squared-up contact, "
                            "ball exit direction closely mirrors bat path, filtering out mishits.")

                col_a1, col_a2, col_a3, col_a4 = st.columns(4)
                with col_a1:
                    st.metric("Attack Angle", f"{attack_angle:+.1f}°",
                              help="Median LA on hard-hit balls — best single proxy for bat path angle")
                with col_a2:
                    st.metric("Mid-Zone Baseline", f"{mid_zone_aa:+.1f}°",
                              help="Attack angle at 2.5ft pitch height from regression")
                with col_a3:
                    st.metric("Path Adjust", f"{path_adjust:+.1f}°/ft",
                              help="How much bat path changes per foot of pitch height")
                with col_a4:
                    delta_aa = attack_angle - avg_la_all
                    st.metric("Hard-Hit vs All LA", f"{delta_aa:+.1f}°",
                              help=f"Hard-hit median ({attack_angle:+.1f}°) minus all-contact median ({avg_la_all:+.1f}°)")

                st.markdown(
                    f'<div style="padding:12px 16px;background:white;border-radius:8px;border-left:5px solid {swing_color};'
                    f'border:1px solid #eee;margin:8px 0;">'
                    f'<span style="font-size:18px;font-weight:900;color:{swing_color} !important;">{swing_type}</span>'
                    f'<div style="font-size:13px;color:#333 !important;margin-top:4px;">{swing_desc}</div>'
                    f'</div>', unsafe_allow_html=True)

                # ═══ SECTION 2: Bat Speed & Contact Depth ═══
                section_header("Bat Speed & Contact Depth")
                st.caption("Bat speed from collision physics: (EV − COR×PitchSpeed) / (1+COR), COR≈0.45. "
                            "Contact depth from Effective Velo − Release Speed: positive = out front, negative = deep.")
                col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                with col_b1:
                    if not pd.isna(bat_speed_avg):
                        st.metric("Bat Speed (est.)", f"{bat_speed_avg:.1f} mph",
                                  help="(EV − 0.45 × PitchSpeed) / 1.45 on hard-hit balls")
                    else:
                        st.metric("Bat Speed (est.)", "—")
                with col_b2:
                    if not pd.isna(bat_speed_max):
                        st.metric("Max Bat Speed", f"{bat_speed_max:.1f} mph")
                    else:
                        st.metric("Max Bat Speed", "—")
                with col_b3:
                    if not pd.isna(bat_speed_pctile):
                        st.metric("DB Percentile", f"{bat_speed_pctile:.0f}th",
                                  help="Where this hitter's avg bat speed ranks among all hitters in the database")
                    else:
                        st.metric("DB Percentile", "—")
                with col_b4:
                    if not pd.isna(contact_depth):
                        st.metric("Contact Depth", f"{contact_depth:+.1f} mph",
                                  help="EffectiveVelo − RelSpeed. Positive = out front, negative = deep")
                    else:
                        st.metric("Contact Depth", "—")

                if depth_label:
                    depth_color = "#fe6100" if depth_label == "Out Front" else ("#2ca02c" if depth_label == "Neutral" else "#1f77b4")
                    st.markdown(
                        f'<div style="padding:8px 14px;background:white;border-radius:8px;border-left:4px solid {depth_color};'
                        f'border:1px solid #eee;margin:6px 0;">'
                        f'<span style="font-weight:700;color:{depth_color} !important;">{depth_label}</span> — '
                        f'<span style="font-size:13px;color:#333 !important;">{depth_desc}</span>'
                        f'</div>', unsafe_allow_html=True)

                # ═══ SECTION 3: Attack Angle by Pitch Type ═══
                section_header("Attack Angle by Pitch Type")
                st.caption("Median LA on hard-hit balls per pitch type — includes bat speed and contact depth per type")
                aa_pt_rows = []
                for pt in sorted(hard_hit["TaggedPitchType"].dropna().unique()):
                    pt_hh = hard_hit[hard_hit["TaggedPitchType"] == pt]
                    pt_all = all_hit[all_hit["TaggedPitchType"] == pt]
                    if len(pt_hh) < 3:
                        continue
                    row = {
                        "Pitch Type": pt,
                        "Hard-Hit n": len(pt_hh),
                        "Total BIP": len(pt_all),
                        "Attack Angle": f"{pt_hh['Angle'].median():+.1f}°",
                        "All LA": f"{pt_all['Angle'].median():+.1f}°",
                        "Hard-Hit EV": f"{pt_hh['ExitSpeed'].mean():.1f}",
                        "Barrel%": f"{int(is_barrel_mask(pt_all).sum()) / max(len(pt_all), 1) * 100:.0f}%",
                    }
                    if "BatSpeedProxy" in pt_hh.columns and pt_hh["BatSpeedProxy"].notna().any():
                        row["Bat Speed"] = f"{pt_hh['BatSpeedProxy'].mean():.1f}"
                    if "ContactDepth" in pt_hh.columns and pt_hh["ContactDepth"].notna().any():
                        row["Depth"] = f"{pt_hh['ContactDepth'].mean():+.1f}"
                    aa_pt_rows.append(row)
                if aa_pt_rows:
                    st.dataframe(pd.DataFrame(aa_pt_rows).set_index("Pitch Type"), use_container_width=True)

                # ═══ SECTION 4: Visualizations ═══
                col_aa_dist, col_aa_height = st.columns(2)
                with col_aa_dist:
                    section_header("Hard-Hit LA Distribution")
                    fig_aa = go.Figure()
                    fig_aa.add_trace(go.Histogram(
                        x=all_hit["Angle"], nbinsx=30,
                        marker_color="#ccc", opacity=0.5, name="All Contact",
                    ))
                    fig_aa.add_trace(go.Histogram(
                        x=hard_hit["Angle"], nbinsx=25,
                        marker_color=swing_color, opacity=0.85, name=f"Hard-Hit (≥{ev_75:.0f} mph)",
                    ))
                    fig_aa.add_vline(x=attack_angle, line_dash="solid", line_color="#1a1a2e",
                                     annotation_text=f"Attack: {attack_angle:+.1f}°", annotation_position="top right")
                    fig_aa.add_vline(x=avg_la_all, line_dash="dash", line_color="#888",
                                     annotation_text=f"All LA: {avg_la_all:+.1f}°", annotation_position="top left")
                    fig_aa.update_layout(**CHART_LAYOUT, height=300, xaxis_title="Launch Angle (°)",
                                          yaxis_title="Count", barmode="overlay",
                                          legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"))
                    st.plotly_chart(fig_aa, use_container_width=True)

                with col_aa_height:
                    section_header("Bat Path vs Pitch Height")
                    st.caption(f"Slope = {path_adjust:+.1f}°/ft | R² = {v_r**2:.2f} | "
                               f"At 2.5ft → {mid_zone_aa:+.1f}°")
                    fig_aa_h = go.Figure()
                    fig_aa_h.add_trace(go.Scatter(
                        x=all_hit["PlateLocHeight"], y=all_hit["Angle"],
                        mode="markers", marker=dict(size=4, color="#ccc", opacity=0.3),
                        name="All Contact",
                    ))
                    fig_aa_h.add_trace(go.Scatter(
                        x=hard_hit["PlateLocHeight"], y=hard_hit["Angle"],
                        mode="markers",
                        marker=dict(size=7, color=hard_hit["ExitSpeed"],
                                    colorscale=[[0, "#fe6100"], [1, "#d22d49"]],
                                    cmin=ev_75, cmax=hard_hit["ExitSpeed"].max(), showscale=True,
                                    colorbar=dict(title="EV", len=0.6),
                                    line=dict(width=0.3, color="white")),
                        name=f"Hard-Hit (≥{ev_75:.0f})",
                        hovertemplate="Height: %{x:.2f}ft<br>LA: %{y:.1f}°<br>EV: %{marker.color:.1f}<extra></extra>",
                    ))
                    x_line = np.linspace(hard_hit["PlateLocHeight"].min(), hard_hit["PlateLocHeight"].max(), 50)
                    fig_aa_h.add_trace(go.Scatter(
                        x=x_line, y=v_int + v_slope * x_line, mode="lines",
                        line=dict(color="#1a1a2e", width=2.5), name="Bat Path Fit",
                    ))
                    fig_aa_h.add_hline(y=0, line_dash="dot", line_color="#ccc")
                    fig_aa_h.update_layout(**CHART_LAYOUT, height=300,
                                            xaxis_title="Pitch Height (ft)", yaxis_title="Launch Angle (°)",
                                            legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"))
                    st.plotly_chart(fig_aa_h, use_container_width=True)

                # ── Horizontal path + Bat speed by height ──
                col_horz, col_bs_h = st.columns(2)
                with col_horz:
                    section_header("Horizontal Bat Path")
                    if not pd.isna(horz_center) and len(h_data) >= 5:
                        st.caption(f"Direction vs plate location on hard-hit balls | Slope = {horz_adjust:+.1f}°/ft | R² = {horz_r2:.2f}")
                        fig_hp = go.Figure()
                        fig_hp.add_trace(go.Scatter(
                            x=h_data["PlateLocSide"], y=h_data["Direction"],
                            mode="markers",
                            marker=dict(size=6, color=h_data["ExitSpeed"],
                                        colorscale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                                        cmin=ev_75, cmax=h_data["ExitSpeed"].max(), showscale=True,
                                        colorbar=dict(title="EV", len=0.6)),
                            hovertemplate="Side: %{x:.2f}<br>Dir: %{y:.1f}°<br>EV: %{marker.color:.1f}<extra></extra>",
                            name="Hard-Hit",
                        ))
                        hx = np.linspace(h_data["PlateLocSide"].min(), h_data["PlateLocSide"].max(), 50)
                        fig_hp.add_trace(go.Scatter(
                            x=hx, y=h_int + h_slope * hx, mode="lines",
                            line=dict(color="#1a1a2e", width=2.5), name="Trend",
                        ))
                        fig_hp.add_hline(y=0, line_dash="dot", line_color="#ccc", annotation_text="Center Field")
                        fig_hp.update_layout(**CHART_LAYOUT, height=300,
                                              xaxis_title="Plate Location (Side)", yaxis_title="Spray Direction (°)",
                                              legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"))
                        st.plotly_chart(fig_hp, use_container_width=True)
                        # Pull tendency note
                        if _bs == "Right":
                            pull_dir = "negative" if horz_center < -10 else ("positive/oppo" if horz_center > 10 else "center")
                        else:
                            pull_dir = "positive" if horz_center > 10 else ("negative/oppo" if horz_center < -10 else "center")
                        st.caption(f"Center-plate spray direction: {horz_center:+.1f}° — tendency: **{pull_dir}**")
                    else:
                        st.info("Not enough hard-hit data with spray direction.")

                with col_bs_h:
                    section_header("Bat Speed by Pitch Height")
                    if "BatSpeedProxy" in hard_hit.columns and hard_hit["BatSpeedProxy"].notna().sum() >= 5:
                        # Bin by pitch height thirds
                        try:
                            height_bins = pd.qcut(hard_hit["PlateLocHeight"], q=3, labels=["Low", "Mid", "High"], duplicates="drop")
                        except ValueError:
                            height_bins = pd.cut(hard_hit["PlateLocHeight"], bins=3, labels=["Low", "Mid", "High"])
                        bs_by_h = hard_hit.groupby(height_bins, observed=True)["BatSpeedProxy"].agg(["mean", "count"]).reset_index()
                        bs_by_h.columns = ["Zone", "Avg Bat Speed", "n"]
                        fig_bs = go.Figure()
                        colors = ["#1f77b4", "#2ca02c", "#d22d49"]
                        for i, (_, row) in enumerate(bs_by_h.iterrows()):
                            fig_bs.add_trace(go.Bar(
                                x=[row["Zone"]], y=[row["Avg Bat Speed"]],
                                marker_color=colors[i % 3], name=row["Zone"],
                                text=f"{row['Avg Bat Speed']:.1f}", textposition="outside",
                                hovertemplate=f"{row['Zone']}: {row['Avg Bat Speed']:.1f} mph (n={row['n']})<extra></extra>",
                            ))
                        fig_bs.update_layout(**CHART_LAYOUT, height=300, yaxis_title="Bat Speed (mph)",
                                              showlegend=False, yaxis=dict(range=[
                                                  max(0, bs_by_h["Avg Bat Speed"].min() - 5),
                                                  bs_by_h["Avg Bat Speed"].max() + 3]))
                        st.plotly_chart(fig_bs, use_container_width=True)
                        best_zone = bs_by_h.loc[bs_by_h["Avg Bat Speed"].idxmax(), "Zone"]
                        st.caption(f"Fastest bat speed in the **{best_zone}** third of the zone")
                    else:
                        st.info("Not enough data for bat speed by height analysis.")

                # ── Contact Depth by Pitch Type ──
                if "ContactDepth" in hard_hit.columns and hard_hit["ContactDepth"].notna().sum() >= 5:
                    col_depth_pt, col_depth_loc = st.columns(2)
                    with col_depth_pt:
                        section_header("Contact Depth by Pitch Type")
                        st.caption("Positive = out front (early), negative = deep (late). Helps identify timing weaknesses.")
                        cd_rows = []
                        for pt in sorted(hard_hit["TaggedPitchType"].dropna().unique()):
                            pt_cd = hard_hit[hard_hit["TaggedPitchType"] == pt]["ContactDepth"]
                            if len(pt_cd) < 3:
                                continue
                            cd_rows.append({"Pitch Type": pt, "Avg Depth": pt_cd.mean(), "n": len(pt_cd)})
                        if cd_rows:
                            cd_df = pd.DataFrame(cd_rows).sort_values("Avg Depth", ascending=False)
                            fig_cd = go.Figure()
                            fig_cd.add_trace(go.Bar(
                                x=cd_df["Pitch Type"], y=cd_df["Avg Depth"],
                                marker_color=[("#fe6100" if d > 0 else "#1f77b4") for d in cd_df["Avg Depth"]],
                                text=[f"{d:+.1f}" for d in cd_df["Avg Depth"]], textposition="outside",
                            ))
                            fig_cd.add_hline(y=0, line_dash="solid", line_color="#333")
                            fig_cd.update_layout(**CHART_LAYOUT, height=300, yaxis_title="Contact Depth (mph)",
                                                  showlegend=False)
                            st.plotly_chart(fig_cd, use_container_width=True)

                    with col_depth_loc:
                        section_header("Contact Depth: Inside vs Outside")
                        st.caption("Does the hitter get out front on inside pitches and stay deep on outside?")
                        if _bs == "Right":
                            in_cd = hard_hit[hard_hit["PlateLocSide"] < -0.28]["ContactDepth"]
                            out_cd = hard_hit[hard_hit["PlateLocSide"] > 0.28]["ContactDepth"]
                        else:
                            in_cd = hard_hit[hard_hit["PlateLocSide"] > 0.28]["ContactDepth"]
                            out_cd = hard_hit[hard_hit["PlateLocSide"] < -0.28]["ContactDepth"]
                        if len(in_cd) >= 3 and len(out_cd) >= 3:
                            fig_io = go.Figure()
                            fig_io.add_trace(go.Bar(x=["Inside"], y=[in_cd.mean()],
                                                     marker_color="#fe6100", text=f"{in_cd.mean():+.1f}",
                                                     textposition="outside", name="Inside"))
                            fig_io.add_trace(go.Bar(x=["Outside"], y=[out_cd.mean()],
                                                     marker_color="#1f77b4", text=f"{out_cd.mean():+.1f}",
                                                     textposition="outside", name="Outside"))
                            fig_io.add_hline(y=0, line_dash="solid", line_color="#333")
                            fig_io.update_layout(**CHART_LAYOUT, height=300, yaxis_title="Contact Depth (mph)",
                                                  showlegend=False)
                            st.plotly_chart(fig_io, use_container_width=True)
                            diff = in_cd.mean() - out_cd.mean()
                            if diff > 0.5:
                                st.caption("Gets out front on inside pitches — good inside coverage, may pull off outside")
                            elif diff < -0.5:
                                st.caption("Deeper on inside pitches — may be late on inside velocity")
                            else:
                                st.caption("Consistent contact depth inside-to-outside — balanced timing")
                        else:
                            st.info("Not enough inside/outside data.")

            else:
                st.info("Not enough InPlay pitches with LA + EV data (need 10+).")

            # ── Barrel Zone Map ──
            section_header("Barrel Path — EV Heatmap")
            st.caption("Where the barrel sweeps through the zone. High EV = barrel center. Low EV = handle/cap contact.")
            if len(inplay_ev) >= 10:
                col_barrel, col_whiff_path = st.columns(2)
                with col_barrel:
                    fig_barrel = go.Figure()
                    fig_barrel.add_trace(go.Histogram2dContour(
                        x=inplay_ev["PlateLocSide"], y=inplay_ev["PlateLocHeight"],
                        z=inplay_ev["ExitSpeed"],
                        histfunc="avg",
                        colorscale=[[0, "#1f77b4"], [0.3, "#f7f7f7"], [0.6, "#fe6100"], [1, "#d22d49"]],
                        contours=dict(showlabels=False),
                        nbinsx=12, nbinsy=12,
                        showscale=True,
                        colorbar=dict(title="Avg EV", len=0.8),
                    ))
                    barrels = inplay_ev[is_barrel_mask(inplay_ev)] if "Angle" in inplay_ev.columns else pd.DataFrame()
                    if len(barrels) > 0:
                        fig_barrel.add_trace(go.Scatter(
                            x=barrels["PlateLocSide"], y=barrels["PlateLocHeight"],
                            mode="markers", marker=dict(size=10, color="#d22d49", symbol="star",
                                                         line=dict(width=1, color="white")),
                            name="Barrels", showlegend=True,
                        ))
                    add_strike_zone(fig_barrel)
                    fig_barrel.update_layout(**CHART_LAYOUT, height=420,
                                              xaxis=dict(range=[-2.5, 2.5], title="Horizontal", scaleanchor="y"),
                                              yaxis=dict(range=[0, 5], title="Vertical"),
                                              title="Contact Quality (EV) by Location",
                                              legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"))
                    st.plotly_chart(fig_barrel, use_container_width=True)

                with col_whiff_path:
                    section_header("Swing & Miss Zones")
                    st.caption("Where the bat path has holes — these are the gaps in the swing plane")
                    wh_loc = whiffs.dropna(subset=["PlateLocSide", "PlateLocHeight"])
                    if len(wh_loc) >= 5:
                        fig_whiff = go.Figure()
                        fig_whiff.add_trace(go.Histogram2dContour(
                            x=wh_loc["PlateLocSide"], y=wh_loc["PlateLocHeight"],
                            colorscale=[[0, "rgba(255,255,255,0)"], [0.3, "rgba(210,45,73,0.2)"],
                                        [0.7, "rgba(210,45,73,0.5)"], [1, "rgba(210,45,73,0.9)"]],
                            contours=dict(showlabels=False),
                            nbinsx=10, nbinsy=10,
                            showscale=True,
                            colorbar=dict(title="Whiff Density", len=0.8),
                        ))
                        con_loc = contacts.dropna(subset=["PlateLocSide", "PlateLocHeight"])
                        if len(con_loc) > 0:
                            fig_whiff.add_trace(go.Scatter(
                                x=con_loc["PlateLocSide"], y=con_loc["PlateLocHeight"],
                                mode="markers", marker=dict(size=3, color="#2ca02c", opacity=0.3),
                                name="Contact", showlegend=True,
                            ))
                        add_strike_zone(fig_whiff)
                        fig_whiff.update_layout(**CHART_LAYOUT, height=420,
                                                  xaxis=dict(range=[-2.5, 2.5], title="Horizontal", scaleanchor="y"),
                                                  yaxis=dict(range=[0, 5], title="Vertical"),
                                                  title="Whiff Density vs Contact Points",
                                                  legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"))
                        st.plotly_chart(fig_whiff, use_container_width=True)
                    else:
                        st.info("Not enough whiff location data.")

            # ── Swing Decisions Map ──
            section_header("Swing Decision Map")
            st.caption("Where this batter swings vs takes — reveals their actual decision zone vs the rule-book strike zone")
            sw_loc = swings.dropna(subset=["PlateLocSide", "PlateLocHeight"])
            takes = bdf[bdf["PitchCall"].isin(["BallCalled", "StrikeCalled"])].dropna(
                subset=["PlateLocSide", "PlateLocHeight"])
            if len(sw_loc) >= 10 and len(takes) >= 10:
                col_dec1, col_dec2 = st.columns(2)
                with col_dec1:
                    fig_dec = go.Figure()
                    fig_dec.add_trace(go.Scatter(
                        x=sw_loc["PlateLocSide"], y=sw_loc["PlateLocHeight"],
                        mode="markers", marker=dict(size=5, color="#d22d49", opacity=0.5),
                        name="Swing",
                    ))
                    fig_dec.add_trace(go.Scatter(
                        x=takes["PlateLocSide"], y=takes["PlateLocHeight"],
                        mode="markers", marker=dict(size=4, color="#2ca02c", opacity=0.3),
                        name="Take",
                    ))
                    add_strike_zone(fig_dec)
                    fig_dec.update_layout(**CHART_LAYOUT, height=400,
                                           xaxis=dict(range=[-2.5, 2.5], title="Horizontal", scaleanchor="y"),
                                           yaxis=dict(range=[0, 5], title="Vertical"),
                                           title="All Pitches: Swing vs Take",
                                           legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"))
                    st.plotly_chart(fig_dec, use_container_width=True)

                with col_dec2:
                    section_header("Swing Probability Contour")
                    all_with_loc = bdf.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
                    all_with_loc["is_swing"] = all_with_loc["PitchCall"].isin(SWING_CALLS).astype(int)
                    fig_prob = go.Figure()
                    fig_prob.add_trace(go.Histogram2dContour(
                        x=all_with_loc["PlateLocSide"], y=all_with_loc["PlateLocHeight"],
                        z=all_with_loc["is_swing"],
                        histfunc="avg",
                        colorscale=[[0, "#2ca02c"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                        contours=dict(showlabels=True, labelfont=dict(size=10)),
                        nbinsx=12, nbinsy=12,
                        showscale=True,
                        colorbar=dict(title="P(Swing)", len=0.8),
                    ))
                    add_strike_zone(fig_prob)
                    fig_prob.update_layout(**CHART_LAYOUT, height=400,
                                            xaxis=dict(range=[-2.5, 2.5], title="Horizontal", scaleanchor="y"),
                                            yaxis=dict(range=[0, 5], title="Vertical"),
                                            title="Swing Probability by Location")
                    st.plotly_chart(fig_prob, use_container_width=True)

            # ── Swing Path Summary ──
            section_header("Swing Path Profile")
            high_swings = swings[swings["PlateLocHeight"] > 2.83] if "PlateLocHeight" in swings.columns else pd.DataFrame()
            low_swings = swings[swings["PlateLocHeight"] < 2.17] if "PlateLocHeight" in swings.columns else pd.DataFrame()
            high_whiff_rate = len(high_swings[high_swings["PitchCall"] == "StrikeSwinging"]) / max(len(high_swings), 1) * 100
            low_whiff_rate = len(low_swings[low_swings["PitchCall"] == "StrikeSwinging"]) / max(len(low_swings), 1) * 100
            high_ev = inplay_ev[inplay_ev["PlateLocHeight"] > 2.83]["ExitSpeed"].mean() if len(inplay_ev[inplay_ev["PlateLocHeight"] > 2.83]) >= 3 else np.nan
            low_ev = inplay_ev[inplay_ev["PlateLocHeight"] < 2.17]["ExitSpeed"].mean() if len(inplay_ev[inplay_ev["PlateLocHeight"] < 2.17]) >= 3 else np.nan

            insights = []
            if not pd.isna(high_ev) and not pd.isna(low_ev):
                if high_ev > low_ev + 3:
                    insights.append(f"Barrel sits **high** — {high_ev:.1f} mph (up) vs {low_ev:.1f} mph (down). Swing plane is elevated.")
                elif low_ev > high_ev + 3:
                    insights.append(f"Barrel sits **low** — {low_ev:.1f} mph (down) vs {high_ev:.1f} mph (up). Swing plane is depressed.")
                else:
                    insights.append(f"Even vertical barrel coverage ({high_ev:.1f} high vs {low_ev:.1f} low).")
            if high_whiff_rate > low_whiff_rate + 10:
                insights.append(f"Vulnerable **up** ({high_whiff_rate:.0f}% whiff high vs {low_whiff_rate:.0f}% low) — bat path sits below high fastball plane.")
            elif low_whiff_rate > high_whiff_rate + 10:
                insights.append(f"Vulnerable **down** ({low_whiff_rate:.0f}% whiff low vs {high_whiff_rate:.0f}% high) — bat sweeps over breaking balls.")
            inside_ev = inplay_ev[inplay_ev["PlateLocSide"] < -0.28]["ExitSpeed"].mean() if _bs == "Right" else inplay_ev[inplay_ev["PlateLocSide"] > 0.28]["ExitSpeed"].mean()
            outside_ev = inplay_ev[inplay_ev["PlateLocSide"] > 0.28]["ExitSpeed"].mean() if _bs == "Right" else inplay_ev[inplay_ev["PlateLocSide"] < -0.28]["ExitSpeed"].mean()
            if not pd.isna(inside_ev) and not pd.isna(outside_ev):
                if inside_ev > outside_ev + 3:
                    insights.append(f"Stronger **inside** ({inside_ev:.1f} mph) than outside ({outside_ev:.1f} mph) — barrel reaches inside pitch well.")
                elif outside_ev > inside_ev + 3:
                    insights.append(f"Stronger **outside** ({outside_ev:.1f} mph) than inside ({inside_ev:.1f} mph) — extends barrel to outer half.")
            if insights:
                for insight in insights:
                    st.markdown(f"- {insight}")
            else:
                st.info("Not enough zone-split data to generate swing path insights.")



def _game_planning_content(data, pitcher=None, season_filter=None, pdf=None, key_prefix="gp"):
    """Pitch sequencing engine, count leverage analysis, and effective velocity.
    If pitcher/pdf provided, skip the pitcher selector (used from Pitching page)."""

    if pitcher is None or pdf is None:
        # Standalone mode: show pitcher selector
        dav_pitching = filter_davidson(data, role="pitcher")
        if dav_pitching.empty:
            st.warning("No Davidson pitching data found.")
            return

        pitchers = sorted(dav_pitching["Pitcher"].unique())
        col_sel1, col_sel2 = st.columns([2, 1])
        with col_sel1:
            pitcher = st.selectbox("Select Pitcher", pitchers, format_func=display_name, key=f"{key_prefix}_pitcher")
        with col_sel2:
            seasons = sorted(dav_pitching["Season"].dropna().unique())
            season_filter = st.multiselect("Season", seasons, default=seasons, key=f"{key_prefix}_season")

        pdf = dav_pitching[dav_pitching["Pitcher"] == pitcher].copy()
        if season_filter:
            pdf = pdf[pdf["Season"].isin(season_filter)]
        if len(pdf) < 30:
            st.warning("Not enough pitches (need 30+).")
            return
        pdf = filter_minor_pitches(pdf)

        jersey = JERSEY.get(pitcher, "")
        pos = POSITION.get(pitcher, "")
        throws = safe_mode(pdf["PitcherThrows"], "")
        t_str = {"Right": "R", "Left": "L"}.get(throws, throws)

        player_header(pitcher, jersey, pos,
                      f"{pos} | Throws: {t_str} | Davidson Wildcats",
                      f"{len(pdf):,} pitches | Seasons: {', '.join(str(int(s)) for s in sorted(pdf['Season'].dropna().unique()))}")

    tab_seq, tab_count, tab_effv = st.tabs(["Sequencing + Tunnels", "Count Leverage", "Effective Velocity"])

    # ─── Tab: Pitch Sequencing + Tunnel Engine ──────────────
    with tab_seq:
        section_header("Pitch Sequencing + Tunnel Engine")
        st.caption("Sequence effectiveness combined with physics-based tunnel analysis — the best sequences are ones that tunnel well AND produce whiffs")

        # Compute tunnel scores for this pitcher
        tunnel_pop = build_tunnel_population_pop()
        tunnel_df = _compute_tunnel_score(pdf, tunnel_pop=tunnel_pop)
        tunnel_lookup = {}
        if not tunnel_df.empty:
            for _, tr in tunnel_df.iterrows():
                tunnel_lookup[(tr["Pitch A"], tr["Pitch B"])] = tr
                tunnel_lookup[(tr["Pitch B"], tr["Pitch A"])] = tr

        # Build sequence pairs
        sort_cols = [c for c in ["GameID", "PAofInning", "Inning", "PitchNo"] if c in pdf.columns]
        if len(sort_cols) >= 2:
            sdf = pdf.sort_values(sort_cols).copy()
            pa_cols = [c for c in ["GameID", "PAofInning", "Inning"] if c in pdf.columns]
            sdf["NextPitch"] = sdf.groupby(pa_cols)["TaggedPitchType"].shift(-1)
            sdf["NextCall"] = sdf.groupby(pa_cols)["PitchCall"].shift(-1)
            sdf["NextEV"] = sdf.groupby(pa_cols)["ExitSpeed"].shift(-1)
            pairs = sdf.dropna(subset=["NextPitch"])

            # ── Tunnel Overview Cards ──
            if not tunnel_df.empty:
                section_header("Tunnel Grades for Pitch Pairs")
                st.caption("Physics-based: pitches that look identical at the commit point (~280ms before plate) but diverge at the plate")
                grade_colors = {"A": "#2ca02c", "B": "#7cb342", "C": "#f7c631", "D": "#fe6100", "F": "#d22d49"}
                tun_cols = st.columns(min(len(tunnel_df), 5))
                for idx, (_, tr) in enumerate(tunnel_df.head(5).iterrows()):
                    gc = grade_colors.get(tr["Grade"], "#aaa")
                    with tun_cols[idx % len(tun_cols)]:
                        st.markdown(
                            f'<div style="text-align:center;padding:10px;background:white;border-radius:8px;'
                            f'border:2px solid {gc};margin:2px;">'
                            f'<div style="font-size:28px;font-weight:900;color:{gc} !important;">{tr["Grade"]}</div>'
                            f'<div style="font-size:12px;font-weight:700;color:#1a1a2e !important;">'
                            f'{tr["Pitch A"]} ↔ {tr["Pitch B"]}</div>'
                            f'<div style="font-size:11px;color:#666 !important;">Score: {tr["Tunnel Score"]:.0f} | '
                            f'Commit: {tr["Commit Sep (in)"]:.1f}″ | Plate: {tr["Plate Sep (in)"]:.1f}″</div>'
                            f'</div>', unsafe_allow_html=True)

            # ── Combined Matrix: Whiff% with Tunnel Grade overlay ──
            section_header("Sequence + Tunnel Matrix")
            st.caption("Cells = Whiff% on next pitch. Border color = tunnel grade for that pair. Best combos have high whiff% AND strong tunnel.")

            pitch_types = sorted(pdf["TaggedPitchType"].dropna().unique())
            matrix_data = np.full((len(pitch_types), len(pitch_types)), np.nan)
            matrix_annot = [['' for _ in pitch_types] for _ in pitch_types]
            matrix_n = [[0 for _ in pitch_types] for _ in pitch_types]

            for i, pt_a in enumerate(pitch_types):
                for j, pt_b in enumerate(pitch_types):
                    pair = pairs[(pairs["TaggedPitchType"] == pt_a) & (pairs["NextPitch"] == pt_b)]
                    if len(pair) < 25:
                        continue
                    sw = pair[pair["NextCall"].isin(SWING_CALLS)]
                    wh = pair[pair["NextCall"] == "StrikeSwinging"]
                    if len(sw) > 0:
                        whiff_pct = len(wh) / len(sw) * 100
                        matrix_data[i, j] = whiff_pct
                        tn = tunnel_lookup.get((pt_a, pt_b))
                        grade_tag = f" [{tn['Grade']}]" if tn is not None else ""
                        matrix_annot[i][j] = f"{whiff_pct:.0f}%{grade_tag}\n({len(pair)})"
                    matrix_n[i][j] = len(pair)

            fig_matrix = go.Figure(data=go.Heatmap(
                z=matrix_data, text=matrix_annot, texttemplate="%{text}",
                x=pitch_types, y=pitch_types,
                colorscale=[[0, "#f7f7f7"], [0.3, "#f7c631"], [0.6, "#fe6100"], [1, "#d22d49"]],
                zmin=0, zmax=50, showscale=True,
                colorbar=dict(title="Whiff%", len=0.8),
            ))
            fig_matrix.update_layout(**CHART_LAYOUT, height=max(300, len(pitch_types) * 60 + 60),
                                      xaxis_title="Next Pitch (B)", yaxis_title="Current Pitch (A)",
                                      xaxis=dict(side="bottom"))
            st.plotly_chart(fig_matrix, use_container_width=True)

            # ── Build full sequence rows with tunnel data ──
            seq_rows = []
            for i, pt_a in enumerate(pitch_types):
                for j, pt_b in enumerate(pitch_types):
                    pair = pairs[(pairs["TaggedPitchType"] == pt_a) & (pairs["NextPitch"] == pt_b)]
                    if len(pair) < 25:
                        continue
                    sw = pair[pair["NextCall"].isin(SWING_CALLS)]
                    wh = pair[pair["NextCall"] == "StrikeSwinging"]
                    ct = pair[pair["NextCall"].isin(CONTACT_CALLS)]
                    bt = pair[pair["NextCall"] == "InPlay"].dropna(subset=["NextEV"])
                    csw = pair[pair["NextCall"].isin(["StrikeSwinging", "StrikeCalled"])]
                    whiff_pct = len(wh) / max(len(sw), 1) * 100 if len(sw) > 0 else 0
                    tn = tunnel_lookup.get((pt_a, pt_b))
                    tunnel_score = tn["Tunnel Score"] if tn is not None else np.nan
                    tunnel_grade = tn["Grade"] if tn is not None else "-"
                    # Composite: 40% tunnel + 30% whiff + 20% EV + 10% CSW
                    whiff_norm = min(whiff_pct / 40.0, 1.0) * 100
                    tunnel_norm = tunnel_score if not pd.isna(tunnel_score) else 50
                    ev_val = bt["NextEV"].mean() if len(bt) > 0 else np.nan
                    ev_norm = max(0, min(100, (105 - ev_val) / 25 * 100)) if not pd.isna(ev_val) else 50
                    csw_pct = len(csw) / len(pair) * 100
                    csw_norm = min(csw_pct / 35.0, 1.0) * 100
                    combo_score = tunnel_norm * 0.40 + whiff_norm * 0.30 + ev_norm * 0.20 + csw_norm * 0.10
                    seq_rows.append({
                        "Sequence": f"{pt_a} → {pt_b}",
                        "Count": len(pair),
                        "Swing%": len(sw) / len(pair) * 100,
                        "Whiff%": whiff_pct,
                        "CSW%": len(csw) / len(pair) * 100,
                        "Avg EV": ev_val,
                        "Tunnel": tunnel_grade,
                        "Tunnel Score": tunnel_score,
                        "Combo Score": round(combo_score, 1),
                        "_whiff": whiff_pct,
                        "_combo": combo_score,
                    })

            if seq_rows:
                seq_df = pd.DataFrame(seq_rows).sort_values("_combo", ascending=False)

                # ── Top Sequences: Combined Ranking ──
                section_header("Top Sequences — Tunnel-Adjusted Ranking")
                st.caption("Combo Score = 50% Whiff Rate + 30% Tunnel Score + 20% Low Contact Quality. Best sequences deceive AND dominate.")
                top_combos = seq_df.head(5)
                for _, row in top_combos.iterrows():
                    gc = grade_colors.get(row["Tunnel"], "#aaa") if not tunnel_df.empty else "#888"
                    ev_str = f" | EV: {row['Avg EV']:.1f}" if not pd.isna(row["Avg EV"]) else ""
                    st.markdown(
                        f'<div style="padding:10px 14px;background:white;border-radius:8px;margin:4px 0;'
                        f'border-left:5px solid {gc};border:1px solid #eee;">'
                        f'<span style="font-size:15px;font-weight:800;color:#1a1a2e !important;">{row["Sequence"]}</span>'
                        f'<span style="display:inline-block;margin-left:8px;padding:2px 8px;border-radius:4px;'
                        f'background:{gc};color:white !important;font-size:11px;font-weight:700;">'
                        f'Tunnel: {row["Tunnel"]}</span>'
                        f'<span style="float:right;font-size:14px;font-weight:900;color:#1a1a2e !important;">'
                        f'Combo: {row["Combo Score"]:.0f}</span>'
                        f'<div style="font-size:12px;color:#555 !important;margin-top:2px;">'
                        f'Whiff: {row["Whiff%"]:.0f}% | CSW: {row["CSW%"]:.0f}%{ev_str} | n={row["Count"]}</div>'
                        f'</div>', unsafe_allow_html=True)

                st.markdown("")

                # ── Side by side: best whiff vs worst damage ──
                col_best, col_worst = st.columns(2)
                with col_best:
                    section_header("Highest Whiff% Sequences")
                    top5 = seq_df.sort_values("_whiff", ascending=False).head(5)
                    for _, row in top5.iterrows():
                        tn_tag = f' <span style="color:{grade_colors.get(row["Tunnel"], "#aaa")} !important;">[{row["Tunnel"]}]</span>' if row["Tunnel"] != "-" else ""
                        ev_str = f"EV: {row['Avg EV']:.1f} | " if not pd.isna(row["Avg EV"]) else ""
                        st.markdown(
                            f'<div style="padding:8px 12px;background:white;border-radius:6px;margin:4px 0;'
                            f'border-left:4px solid #2ca02c;border:1px solid #eee;">'
                            f'<span style="font-size:14px;font-weight:700;color:#1a1a2e !important;">{row["Sequence"]}{tn_tag}</span>'
                            f'<span style="float:right;font-size:13px;font-weight:800;color:#2ca02c !important;">'
                            f'Whiff: {row["Whiff%"]:.0f}% | CSW: {row["CSW%"]:.0f}%</span>'
                            f'<div style="font-size:11px;color:#666 !important;">{ev_str}n={row["Count"]}</div></div>',
                            unsafe_allow_html=True)
                with col_worst:
                    section_header("Most Damage Sequences")
                    bot5 = seq_df.dropna(subset=["Avg EV"]).sort_values("Avg EV", ascending=False).head(5)
                    for _, row in bot5.iterrows():
                        tn_tag = f' <span style="color:{grade_colors.get(row["Tunnel"], "#aaa")} !important;">[{row["Tunnel"]}]</span>' if row["Tunnel"] != "-" else ""
                        st.markdown(
                            f'<div style="padding:8px 12px;background:white;border-radius:6px;margin:4px 0;'
                            f'border-left:4px solid #d22d49;border:1px solid #eee;">'
                            f'<span style="font-size:14px;font-weight:700;color:#1a1a2e !important;">{row["Sequence"]}{tn_tag}</span>'
                            f'<span style="float:right;font-size:13px;font-weight:800;color:#d22d49 !important;">'
                            f'EV: {row["Avg EV"]:.1f} | Whiff: {row["Whiff%"]:.0f}%</span>'
                            f'<div style="font-size:11px;color:#666 !important;">n={row["Count"]}</div></div>',
                            unsafe_allow_html=True)

                # ── Tunnel Divergence Visualization for Top Pair ──
                if not tunnel_df.empty:
                    section_header("Commit-Point Divergence — Top Tunnel Pair")
                    best_tn = tunnel_df.iloc[0]
                    st.caption(f"How {best_tn['Pitch A']} and {best_tn['Pitch B']} separate in flight — "
                               f"closer at commit = more deception, farther at plate = more movement")

                    # Recompute flight paths for visualization
                    req_cols = ["TaggedPitchType", "RelHeight", "RelSide", "PlateLocHeight", "PlateLocSide",
                                "InducedVertBreak", "HorzBreak", "RelSpeed"]
                    if all(c in pdf.columns for c in req_cols):
                        agg_cols_viz = {
                            "rel_h": ("RelHeight", "mean"), "rel_s": ("RelSide", "mean"),
                            "loc_h": ("PlateLocHeight", "mean"), "loc_s": ("PlateLocSide", "mean"),
                            "ivb": ("InducedVertBreak", "mean"), "hb": ("HorzBreak", "mean"),
                            "velo": ("RelSpeed", "mean"),
                        }
                        if "Extension" in pdf.columns:
                            agg_cols_viz["ext"] = ("Extension", "mean")
                        agg_viz = pdf.groupby("TaggedPitchType").agg(**agg_cols_viz).dropna(subset=["rel_h", "velo"])
                        if "ext" not in agg_viz.columns:
                            agg_viz["ext"] = 6.0

                        pt_a_name, pt_b_name = best_tn["Pitch A"], best_tn["Pitch B"]
                        if pt_a_name in agg_viz.index and pt_b_name in agg_viz.index:
                            MOUND_DIST = 60.5
                            def _viz_flight(row, frac):
                                ext = row.ext if not pd.isna(row.ext) else 6.0
                                actual_dist = MOUND_DIST - ext
                                velo_fps = row.velo * 5280 / 3600
                                t_total = actual_dist / velo_fps
                                t = frac * t_total
                                gravity_drop = 0.5 * 32.17 * t**2
                                ivb_lift = (row.ivb / 12.0) * frac**2
                                y = row.rel_h + (row.loc_h - row.rel_h) * frac - gravity_drop + ivb_lift
                                y_at_1 = row.rel_h + (row.loc_h - row.rel_h) - 0.5 * 32.17 * t_total**2 + (row.ivb / 12.0)
                                y += (row.loc_h - y_at_1) * frac
                                hb_curve = (row.hb / 12.0) * frac**2
                                x = row.rel_s + (row.loc_s - row.rel_s) * frac + hb_curve
                                x_at_1 = row.rel_s + (row.loc_s - row.rel_s) + (row.hb / 12.0)
                                x += (row.loc_s - x_at_1) * frac
                                return x, y

                            checkpoints = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                            labels = ["Release", "20%", "40%", "Commit (60%)", "80%", "Plate"]
                            a_row = agg_viz.loc[pt_a_name]
                            b_row = agg_viz.loc[pt_b_name]
                            seps = []
                            a_xs, a_ys, b_xs, b_ys = [], [], [], []
                            for frac in checkpoints:
                                ax, ay = _viz_flight(a_row, frac)
                                bx, by = _viz_flight(b_row, frac)
                                a_xs.append(ax); a_ys.append(ay)
                                b_xs.append(bx); b_ys.append(by)
                                sep_in = np.sqrt((ay - by)**2 + (ax - bx)**2) * 12
                                seps.append(sep_in)

                            col_div, col_path = st.columns(2)
                            with col_div:
                                # Divergence line chart
                                fig_div = go.Figure()
                                fig_div.add_trace(go.Scatter(
                                    x=labels, y=seps, mode="lines+markers+text",
                                    text=[f"{s:.1f}″" for s in seps],
                                    textposition="top center",
                                    line=dict(color="#1a1a2e", width=3),
                                    marker=dict(size=10, color=[grade_colors.get(best_tn["Grade"], "#aaa")] * len(seps)),
                                    showlegend=False,
                                ))
                                fig_div.add_hline(y=6, line_dash="dash", line_color="#d22d49",
                                                  annotation_text="6″ — Hard to distinguish", annotation_position="top left")
                                fig_div.update_layout(**CHART_LAYOUT, height=350,
                                                      yaxis_title="Separation (inches)",
                                                      title=f"{pt_a_name} vs {pt_b_name} — Flight Separation")
                                st.plotly_chart(fig_div, use_container_width=True)

                            with col_path:
                                # Side view: both trajectories
                                fig_path = go.Figure()
                                clr_a = PITCH_COLORS.get(pt_a_name, "#1f77b4")
                                clr_b = PITCH_COLORS.get(pt_b_name, "#ff7f0e")
                                dist_pts = [0, 0.2 * (MOUND_DIST - 6), 0.4 * (MOUND_DIST - 6),
                                            0.6 * (MOUND_DIST - 6), 0.8 * (MOUND_DIST - 6),
                                            MOUND_DIST - 6]
                                fig_path.add_trace(go.Scatter(
                                    x=dist_pts, y=a_ys, mode="lines+markers",
                                    name=pt_a_name, line=dict(color=clr_a, width=3),
                                    marker=dict(size=8),
                                ))
                                fig_path.add_trace(go.Scatter(
                                    x=dist_pts, y=b_ys, mode="lines+markers",
                                    name=pt_b_name, line=dict(color=clr_b, width=3),
                                    marker=dict(size=8),
                                ))
                                # Commit point marker
                                fig_path.add_vline(x=0.6 * (MOUND_DIST - 6), line_dash="dot", line_color="#888",
                                                   annotation_text="Commit", annotation_position="top")
                                fig_path.update_layout(**CHART_LAYOUT, height=350,
                                                        xaxis_title="Distance from release (ft)",
                                                        yaxis_title="Height (ft)",
                                                        title="Side View — Pitch Trajectories",
                                                        yaxis=dict(range=[0, 7]),
                                                        legend=dict(x=0.02, y=0.98))
                                st.plotly_chart(fig_path, use_container_width=True)

                            # Diagnosis from tunnel data
                            st.markdown(
                                f'<div style="padding:12px 16px;background:#f0f7ff;border-radius:8px;border:1px solid #cce0ff;">'
                                f'<span style="font-size:13px;font-weight:700;color:#1a1a2e !important;">Tunnel Analysis:</span> '
                                f'<span style="font-size:12px;color:#333 !important;">{best_tn["Diagnosis"]}</span>'
                                f'<br><span style="font-size:12px;font-weight:600;color:#1565c0 !important;">Action: </span>'
                                f'<span style="font-size:12px;color:#333 !important;">{best_tn["Fix"]}</span></div>',
                                unsafe_allow_html=True)

                # Full table with tunnel data
                with st.expander("Full 2-Pitch Sequence + Tunnel Table"):
                    disp_seq = seq_df.drop(columns=["_whiff", "_combo"]).copy()
                    for c in ["Swing%", "Whiff%", "CSW%"]:
                        disp_seq[c] = disp_seq[c].map(lambda x: f"{x:.1f}%")
                    disp_seq["Avg EV"] = disp_seq["Avg EV"].map(lambda x: f"{x:.1f}" if not pd.isna(x) else "-")
                    disp_seq["Tunnel Score"] = disp_seq["Tunnel Score"].map(lambda x: f"{x:.0f}" if not pd.isna(x) else "-")
                    disp_seq = disp_seq.sort_values("Combo Score", ascending=False)
                    disp_seq["Combo Score"] = disp_seq["Combo Score"].map(lambda x: f"{x:.0f}")
                    st.dataframe(disp_seq.set_index("Sequence"), use_container_width=True)

            # ── 3-PITCH & 4-PITCH SEQUENCE ANALYSIS ──────────────
            st.markdown("---")
            section_header("Multi-Pitch Sequence Patterns")
            st.caption("3-pitch and 4-pitch sequences with conditional outcome probabilities — "
                       "what actually happens when this pitcher follows a pattern")

            # Build extended sequence columns from the sorted pitch data
            sdf2 = sdf.copy()
            sdf2["Pitch2"] = sdf2.groupby(pa_cols)["TaggedPitchType"].shift(-1)
            sdf2["Pitch3"] = sdf2.groupby(pa_cols)["TaggedPitchType"].shift(-2)
            sdf2["Pitch4"] = sdf2.groupby(pa_cols)["TaggedPitchType"].shift(-3)
            sdf2["Call1"] = sdf2["PitchCall"]
            sdf2["Call2"] = sdf2.groupby(pa_cols)["PitchCall"].shift(-1)
            sdf2["Call3"] = sdf2.groupby(pa_cols)["PitchCall"].shift(-2)
            sdf2["Call4"] = sdf2.groupby(pa_cols)["PitchCall"].shift(-3)
            sdf2["EV3"] = sdf2.groupby(pa_cols)["ExitSpeed"].shift(-2)
            sdf2["EV4"] = sdf2.groupby(pa_cols)["ExitSpeed"].shift(-3)

            # Outcome classification helper
            def _classify_outcome(call, ev=np.nan):
                if call == "StrikeSwinging":
                    return "Whiff"
                elif call == "StrikeCalled":
                    return "Called Strike"
                elif call in ("BallCalled", "HitByPitch", "BallinDirt", "BallIntentional"):
                    return "Ball"
                elif call in ("FoulBall", "FoulBallNotFieldable", "FoulBallFieldable"):
                    return "Foul"
                elif call == "InPlay":
                    if not pd.isna(ev) and ev < 75:
                        return "Weak Contact"
                    elif not pd.isna(ev) and ev >= 95:
                        return "Hard Contact"
                    else:
                        return "In Play"
                return "Other"

            tab_3p, tab_4p, tab_tree = st.tabs(["3-Pitch Sequences", "4-Pitch Sequences", "Sequence Trees"])

            # ── 3-Pitch Sequences ──
            with tab_3p:
                section_header("3-Pitch Sequence Analysis")
                st.markdown("""
                Analyzes every **3-consecutive-pitch window** within each at-bat.
                For each sequence (e.g. Slider → Fastball → Changeup), the table shows how effective the **3rd pitch** is —
                whiff rate, called-strike rate, contact quality — and how outcomes change depending on whether the **1st pitch was a strike or ball**.
                Use this to find which 3-pitch patterns generate the most swings-and-misses or weak contact.
                """)

                seq3 = sdf2.dropna(subset=["Pitch2", "Pitch3", "Call3"]).copy()
                if len(seq3) < 20:
                    st.info("Not enough 3-pitch sequences (need 20+).")
                else:
                    # Build 3-pitch sequence stats
                    seq3["Seq3"] = seq3["TaggedPitchType"] + " → " + seq3["Pitch2"] + " → " + seq3["Pitch3"]
                    seq3_counts = seq3["Seq3"].value_counts()
                    top_seq3 = seq3_counts[seq3_counts >= 25].head(20).index.tolist()

                    if top_seq3:
                        seq3_rows = []
                        for s3 in top_seq3:
                            s3_df = seq3[seq3["Seq3"] == s3]
                            n = len(s3_df)
                            # Outcome on 3rd pitch
                            sw3 = s3_df[s3_df["Call3"].isin(SWING_CALLS)]
                            wh3 = s3_df[s3_df["Call3"] == "StrikeSwinging"]
                            csw3 = s3_df[s3_df["Call3"].isin(["StrikeSwinging", "StrikeCalled"])]
                            ip3 = s3_df[s3_df["Call3"] == "InPlay"]
                            ev3_vals = ip3["EV3"].dropna()
                            # Conditional: what was Call1 outcome?
                            call1_strike = s3_df[s3_df["Call1"].isin(["StrikeCalled", "StrikeSwinging",
                                                                       "FoulBall", "FoulBallNotFieldable",
                                                                       "FoulBallFieldable", "InPlay"])]
                            pct_strike1 = len(call1_strike) / max(n, 1) * 100

                            # Tunnel data for the pair transitions
                            pitches = s3.split(" → ")
                            if len(pitches) != 3:
                                continue
                            tn_12 = tunnel_lookup.get((pitches[0], pitches[1]))
                            tn_23 = tunnel_lookup.get((pitches[1], pitches[2]))
                            tunnel_avg = np.nanmean([
                                tn_12["Tunnel Score"] if tn_12 is not None else np.nan,
                                tn_23["Tunnel Score"] if tn_23 is not None else np.nan,
                            ])

                            seq3_rows.append({
                                "Sequence": s3,
                                "n": n,
                                "Strike1%": pct_strike1,
                                "Whiff3%": len(wh3) / max(len(sw3), 1) * 100 if len(sw3) > 0 else 0,
                                "CSW3%": len(csw3) / n * 100,
                                "Swing3%": len(sw3) / n * 100,
                                "Avg EV3": ev3_vals.mean() if len(ev3_vals) > 0 else np.nan,
                                "Weak%": len(ev3_vals[ev3_vals < 75]) / max(len(ev3_vals), 1) * 100 if len(ev3_vals) > 0 else np.nan,
                                "Tunnel Avg": tunnel_avg,
                                "_score": (tunnel_avg if not pd.isna(tunnel_avg) else 50) * 0.4
                                          + (len(wh3) / max(len(sw3), 1) * 100 if len(sw3) > 0 else 0) * 0.3
                                          + (len(csw3) / n * 100) * 0.3,
                            })

                        if seq3_rows:
                            seq3_df = pd.DataFrame(seq3_rows).sort_values("_score", ascending=False)

                            # Top 3-pitch sequences
                            st.markdown("**Top 3-Pitch Sequences (by combined whiff + CSW + tunnel)**")
                            for _, row in seq3_df.head(8).iterrows():
                                tn_str = f"{row['Tunnel Avg']:.0f}" if not pd.isna(row["Tunnel Avg"]) else "-"
                                ev_str = f" | EV: {row['Avg EV3']:.1f}" if not pd.isna(row["Avg EV3"]) else ""
                                weak_str = f" | Weak: {row['Weak%']:.0f}%" if not pd.isna(row["Weak%"]) else ""
                                score_clr = "#2ca02c" if row["_score"] > 50 else "#fe6100" if row["_score"] > 35 else "#d22d49"
                                st.markdown(
                                    f'<div style="padding:8px 14px;background:white;border-radius:6px;margin:3px 0;'
                                    f'border-left:4px solid {score_clr};border:1px solid #eee;">'
                                    f'<span style="font-size:14px;font-weight:800;color:#1a1a2e !important;">'
                                    f'{row["Sequence"]}</span>'
                                    f'<span style="float:right;font-size:12px;font-weight:700;color:{score_clr} !important;">'
                                    f'Score: {row["_score"]:.0f}</span>'
                                    f'<div style="font-size:11px;color:#555 !important;">'
                                    f'n={row["n"]} | Strike1: {row["Strike1%"]:.0f}% | '
                                    f'Whiff3: {row["Whiff3%"]:.0f}% | CSW3: {row["CSW3%"]:.0f}%{ev_str}{weak_str} | '
                                    f'Tunnel: {tn_str}</div></div>', unsafe_allow_html=True)

                            # Conditional outcome breakdown
                            section_header("Conditional Outcomes — \"If Pitch 1 = Strike, then...\"")
                            st.caption("How the 3rd pitch performs when Pitch 1 was a strike vs ball")

                            top3_seq = seq3_df["Sequence"].head(5).tolist()
                            cond_rows = []
                            for s3 in top3_seq:
                                s3_df = seq3[seq3["Seq3"] == s3]
                                # Split by Pitch 1 outcome
                                strike1 = s3_df[s3_df["Call1"].isin(["StrikeCalled", "StrikeSwinging",
                                                                      "FoulBall", "FoulBallNotFieldable",
                                                                      "FoulBallFieldable", "InPlay"])]
                                ball1 = s3_df[~s3_df.index.isin(strike1.index)]
                                for label_c, cdf in [("After Strike", strike1), ("After Ball", ball1)]:
                                    if len(cdf) < 2:
                                        continue
                                    sw = cdf[cdf["Call3"].isin(SWING_CALLS)]
                                    wh = cdf[cdf["Call3"] == "StrikeSwinging"]
                                    ip = cdf[cdf["Call3"] == "InPlay"]
                                    ev_v = ip["EV3"].dropna()
                                    cond_rows.append({
                                        "Sequence": s3,
                                        "Condition": label_c,
                                        "n": len(cdf),
                                        "Swing%": f"{len(sw)/len(cdf)*100:.0f}%",
                                        "Whiff%": f"{len(wh)/max(len(sw),1)*100:.0f}%" if len(sw) > 0 else "-",
                                        "InPlay%": f"{len(ip)/len(cdf)*100:.0f}%",
                                        "Avg EV": f"{ev_v.mean():.1f}" if len(ev_v) > 0 else "-",
                                        "Weak%": f"{len(ev_v[ev_v<75])/max(len(ev_v),1)*100:.0f}%" if len(ev_v) > 0 else "-",
                                    })
                            if cond_rows:
                                st.dataframe(pd.DataFrame(cond_rows).set_index(["Sequence", "Condition"]),
                                             use_container_width=True)

                            # Full table
                            with st.expander("Full 3-Pitch Sequence Table"):
                                disp3 = seq3_df.drop(columns=["_score"]).copy()
                                for c in ["Strike1%", "Whiff3%", "CSW3%", "Swing3%"]:
                                    disp3[c] = disp3[c].map(lambda x: f"{x:.1f}%")
                                disp3["Avg EV3"] = disp3["Avg EV3"].map(lambda x: f"{x:.1f}" if not pd.isna(x) else "-")
                                disp3["Weak%"] = disp3["Weak%"].map(lambda x: f"{x:.0f}%" if not pd.isna(x) else "-")
                                disp3["Tunnel Avg"] = disp3["Tunnel Avg"].map(lambda x: f"{x:.0f}" if not pd.isna(x) else "-")
                                st.dataframe(disp3.set_index("Sequence"), use_container_width=True)

            # ── 4-Pitch Sequences ──
            with tab_4p:
                section_header("4-Pitch Sequence Analysis")
                st.caption("Pitch 1 → Pitch 2 → Pitch 3 → Pitch 4: Deep pattern analysis")

                seq4 = sdf2.dropna(subset=["Pitch2", "Pitch3", "Pitch4", "Call4"]).copy()
                if len(seq4) < 20:
                    st.info("Not enough 4-pitch sequences (need 20+).")
                else:
                    seq4["Seq4"] = (seq4["TaggedPitchType"] + " → " + seq4["Pitch2"] + " → " +
                                    seq4["Pitch3"] + " → " + seq4["Pitch4"])
                    seq4_counts = seq4["Seq4"].value_counts()
                    top_seq4 = seq4_counts[seq4_counts >= 25].head(20).index.tolist()

                    if top_seq4:
                        seq4_rows = []
                        for s4 in top_seq4:
                            s4_df = seq4[seq4["Seq4"] == s4]
                            n = len(s4_df)
                            sw4 = s4_df[s4_df["Call4"].isin(SWING_CALLS)]
                            wh4 = s4_df[s4_df["Call4"] == "StrikeSwinging"]
                            csw4 = s4_df[s4_df["Call4"].isin(["StrikeSwinging", "StrikeCalled"])]
                            ip4 = s4_df[s4_df["Call4"] == "InPlay"]
                            ev4_vals = ip4["EV4"].dropna()

                            # Count strikes in first 3 pitches
                            strike_calls = ["StrikeCalled", "StrikeSwinging", "FoulBall",
                                            "FoulBallNotFieldable", "FoulBallFieldable", "InPlay"]
                            s4_df_c = s4_df.copy()
                            s4_df_c["_str1"] = s4_df_c["Call1"].isin(strike_calls).astype(int)
                            s4_df_c["_str2"] = s4_df_c["Call2"].isin(strike_calls).astype(int)
                            s4_df_c["_str3"] = s4_df_c["Call3"].isin(strike_calls).astype(int)
                            avg_strikes = (s4_df_c["_str1"] + s4_df_c["_str2"] + s4_df_c["_str3"]).mean()

                            pitches = s4.split(" → ")
                            tunnel_scores = []
                            for k in range(len(pitches) - 1):
                                tn = tunnel_lookup.get((pitches[k], pitches[k+1]))
                                if tn is not None:
                                    tunnel_scores.append(tn["Tunnel Score"])
                            tunnel_avg = np.mean(tunnel_scores) if tunnel_scores else np.nan

                            seq4_rows.append({
                                "Sequence": s4,
                                "n": n,
                                "Avg Strikes (1-3)": round(avg_strikes, 1),
                                "Whiff4%": len(wh4) / max(len(sw4), 1) * 100 if len(sw4) > 0 else 0,
                                "CSW4%": len(csw4) / n * 100,
                                "Swing4%": len(sw4) / n * 100,
                                "Avg EV4": ev4_vals.mean() if len(ev4_vals) > 0 else np.nan,
                                "Tunnel Avg": tunnel_avg,
                                "_score": (tunnel_avg if not pd.isna(tunnel_avg) else 50) * 0.4
                                          + (len(wh4) / max(len(sw4), 1) * 100 if len(sw4) > 0 else 0) * 0.3
                                          + (len(csw4) / n * 100) * 0.3,
                            })

                        if seq4_rows:
                            seq4_df = pd.DataFrame(seq4_rows).sort_values("_score", ascending=False)

                            st.markdown("**Top 4-Pitch Sequences**")
                            for _, row in seq4_df.head(8).iterrows():
                                tn_str = f"{row['Tunnel Avg']:.0f}" if not pd.isna(row["Tunnel Avg"]) else "-"
                                ev_str = f" | EV: {row['Avg EV4']:.1f}" if not pd.isna(row["Avg EV4"]) else ""
                                score_clr = "#2ca02c" if row["_score"] > 50 else "#fe6100" if row["_score"] > 35 else "#d22d49"
                                st.markdown(
                                    f'<div style="padding:8px 14px;background:white;border-radius:6px;margin:3px 0;'
                                    f'border-left:4px solid {score_clr};border:1px solid #eee;">'
                                    f'<span style="font-size:14px;font-weight:800;color:#1a1a2e !important;">'
                                    f'{row["Sequence"]}</span>'
                                    f'<span style="float:right;font-size:12px;font-weight:700;color:{score_clr} !important;">'
                                    f'Score: {row["_score"]:.0f}</span>'
                                    f'<div style="font-size:11px;color:#555 !important;">'
                                    f'n={row["n"]} | Strikes(1-3): {row["Avg Strikes (1-3)"]:.1f} | '
                                    f'Whiff4: {row["Whiff4%"]:.0f}% | CSW4: {row["CSW4%"]:.0f}%{ev_str} | '
                                    f'Tunnel: {tn_str}</div></div>', unsafe_allow_html=True)

                            with st.expander("Full 4-Pitch Sequence Table"):
                                disp4 = seq4_df.drop(columns=["_score"]).copy()
                                for c in ["Whiff4%", "CSW4%", "Swing4%"]:
                                    disp4[c] = disp4[c].map(lambda x: f"{x:.1f}%")
                                disp4["Avg EV4"] = disp4["Avg EV4"].map(lambda x: f"{x:.1f}" if not pd.isna(x) else "-")
                                disp4["Tunnel Avg"] = disp4["Tunnel Avg"].map(lambda x: f"{x:.0f}" if not pd.isna(x) else "-")
                                st.dataframe(disp4.set_index("Sequence"), use_container_width=True)
                    else:
                        st.info("No 4-pitch sequences with 3+ occurrences found.")

            # ── Sequence Decision Trees ──
            with tab_tree:
                section_header("Sequence Decision Trees")
                st.caption("Select a starting pitch to see branching probabilities — "
                           "\"After Slider strike → what comes next? What's the outcome?\"")

                # Starting pitch selection
                start_pitch = st.selectbox("Start with pitch:", pitch_types, key=f"{key_prefix}_tree_start")

                # Build the tree from sdf2
                tree_base = sdf2[sdf2["TaggedPitchType"] == start_pitch].copy()
                if len(tree_base) < 10:
                    st.info(f"Not enough at-bats starting with {start_pitch}.")
                else:
                    # Level 1: What outcome on Pitch 1?
                    tree_base["Out1"] = tree_base["Call1"].map(lambda c: "Strike" if c in [
                        "StrikeCalled", "StrikeSwinging", "FoulBall", "FoulBallNotFieldable",
                        "FoulBallFieldable", "InPlay"] else "Ball")

                    section_header(f"After {start_pitch}...")
                    for outcome1 in ["Strike", "Ball"]:
                        o1_df = tree_base[tree_base["Out1"] == outcome1]
                        if len(o1_df) < 3:
                            continue
                        o1_pct = len(o1_df) / len(tree_base) * 100

                        # What pitch comes next?
                        o1_with_p2 = o1_df.dropna(subset=["Pitch2"])
                        if len(o1_with_p2) < 3:
                            continue
                        p2_counts = o1_with_p2["Pitch2"].value_counts()
                        o1_clr = "#2ca02c" if outcome1 == "Strike" else "#d22d49"

                        st.markdown(
                            f'<div style="padding:10px 14px;background:#f8f8f8;border-radius:8px;'
                            f'border-left:5px solid {o1_clr};margin:6px 0;">'
                            f'<span style="font-size:15px;font-weight:800;color:{o1_clr} !important;">'
                            f'{start_pitch} = {outcome1} ({o1_pct:.0f}%)</span></div>',
                            unsafe_allow_html=True)

                        # Level 2 branches
                        branch_rows = []
                        for p2_name in p2_counts.head(4).index:
                            p2_df = o1_with_p2[o1_with_p2["Pitch2"] == p2_name]
                            p2_pct = len(p2_df) / len(o1_with_p2) * 100
                            # Outcome on Pitch 2
                            sw2 = p2_df[p2_df["Call2"].isin(SWING_CALLS)]
                            wh2 = p2_df[p2_df["Call2"] == "StrikeSwinging"]
                            csw2 = p2_df[p2_df["Call2"].isin(["StrikeSwinging", "StrikeCalled"])]
                            ip2 = p2_df[p2_df["Call2"] == "InPlay"]

                            # Level 3: what's the 3rd pitch after this?
                            p2_with_p3 = p2_df.dropna(subset=["Pitch3"])
                            p3_top = p2_with_p3["Pitch3"].value_counts().head(3) if len(p2_with_p3) >= 3 else pd.Series(dtype=float)
                            p3_str = ", ".join([f"{pt} ({ct/len(p2_with_p3)*100:.0f}%)" for pt, ct in p3_top.items()]) if len(p3_top) > 0 else "-"

                            # Outcome summary on pitch 2
                            outcomes = []
                            if len(wh2) > 0:
                                outcomes.append(f"Whiff {len(wh2)/len(sw2)*100:.0f}%" if len(sw2) > 0 else "Whiff -")
                            if len(csw2) > 0:
                                outcomes.append(f"CSW {len(csw2)/len(p2_df)*100:.0f}%")
                            if len(ip2) > 0:
                                ev_avg = sdf2.loc[ip2.index, "ExitSpeed"].dropna().mean() if "ExitSpeed" in sdf2.columns else np.nan
                                outcomes.append(f"InPlay {len(ip2)} ({ev_avg:.0f} EV)" if not pd.isna(ev_avg) else f"InPlay {len(ip2)}")

                            branch_rows.append({
                                "Next Pitch": p2_name,
                                "Frequency": f"{p2_pct:.0f}%",
                                "n": len(p2_df),
                                "P2 Outcome": " | ".join(outcomes) if outcomes else "-",
                                "Then Pitch 3": p3_str,
                            })

                        if branch_rows:
                            bdf_tree = pd.DataFrame(branch_rows)
                            st.dataframe(bdf_tree.set_index("Next Pitch"), use_container_width=True)

                    # Head-to-head comparison
                    section_header("Compare Sequences Head-to-Head")
                    st.caption("Does Slider → Slider → Fastball work better than Slider → Fastball → Slider?")
                    seq3_all = sdf2.dropna(subset=["Pitch2", "Pitch3", "Call3"]).copy()
                    seq3_all["Seq3"] = seq3_all["TaggedPitchType"] + " → " + seq3_all["Pitch2"] + " → " + seq3_all["Pitch3"]
                    avail_seqs = seq3_all["Seq3"].value_counts()
                    avail_seqs = avail_seqs[avail_seqs >= 3].index.tolist()

                    if len(avail_seqs) >= 2:
                        col_cmp1, col_cmp2 = st.columns(2)
                        with col_cmp1:
                            seq_a = st.selectbox("Sequence A", avail_seqs, key=f"{key_prefix}_cmp_a")
                        with col_cmp2:
                            seq_b = st.selectbox("Sequence B", [s for s in avail_seqs if s != seq_a],
                                                 key=f"{key_prefix}_cmp_b")

                        cmp_rows = []
                        for seq_name in [seq_a, seq_b]:
                            cdf = seq3_all[seq3_all["Seq3"] == seq_name]
                            n = len(cdf)
                            sw = cdf[cdf["Call3"].isin(SWING_CALLS)]
                            wh = cdf[cdf["Call3"] == "StrikeSwinging"]
                            csw = cdf[cdf["Call3"].isin(["StrikeSwinging", "StrikeCalled"])]
                            ip = cdf[cdf["Call3"] == "InPlay"]
                            ev_v = ip["EV3"].dropna()
                            weak = ev_v[ev_v < 75] if len(ev_v) > 0 else pd.Series(dtype=float)
                            hard = ev_v[ev_v >= 95] if len(ev_v) > 0 else pd.Series(dtype=float)

                            pitches = seq_name.split(" → ")
                            tscores = []
                            for k in range(len(pitches) - 1):
                                tn = tunnel_lookup.get((pitches[k], pitches[k+1]))
                                if tn is not None:
                                    tscores.append(tn["Tunnel Score"])
                            t_avg = np.mean(tscores) if tscores else np.nan

                            cmp_rows.append({
                                "Sequence": seq_name,
                                "n": n,
                                "Swing%": f"{len(sw)/n*100:.1f}%",
                                "Whiff%": f"{len(wh)/max(len(sw),1)*100:.1f}%" if len(sw) > 0 else "-",
                                "CSW%": f"{len(csw)/n*100:.1f}%",
                                "InPlay": len(ip),
                                "Avg EV": f"{ev_v.mean():.1f}" if len(ev_v) > 0 else "-",
                                "Weak Contact%": f"{len(weak)/max(len(ev_v),1)*100:.0f}%" if len(ev_v) > 0 else "-",
                                "Hard Contact%": f"{len(hard)/max(len(ev_v),1)*100:.0f}%" if len(ev_v) > 0 else "-",
                                "Tunnel Avg": f"{t_avg:.0f}" if not pd.isna(t_avg) else "-",
                            })

                        if cmp_rows:
                            st.dataframe(pd.DataFrame(cmp_rows).set_index("Sequence"), use_container_width=True)

                            # Winner declaration
                            a_row = cmp_rows[0]
                            b_row = cmp_rows[1]
                            a_whiff = float(a_row["Whiff%"].replace("%", "")) if a_row["Whiff%"] != "-" else 0
                            b_whiff = float(b_row["Whiff%"].replace("%", "")) if b_row["Whiff%"] != "-" else 0
                            a_csw = float(a_row["CSW%"].replace("%", "")) if a_row["CSW%"] != "-" else 0
                            b_csw = float(b_row["CSW%"].replace("%", "")) if b_row["CSW%"] != "-" else 0
                            a_tun = float(a_row["Tunnel Avg"]) if a_row["Tunnel Avg"] != "-" else 0
                            b_tun = float(b_row["Tunnel Avg"]) if b_row["Tunnel Avg"] != "-" else 0
                            if a_whiff + a_csw + a_tun * 0.5 > b_whiff + b_csw + b_tun * 0.5:
                                winner = seq_a
                                w_clr = "#2ca02c"
                            elif b_whiff + b_csw + b_tun * 0.5 > a_whiff + a_csw + a_tun * 0.5:
                                winner = seq_b
                                w_clr = "#2ca02c"
                            else:
                                winner = "Tie"
                                w_clr = "#888"
                            if winner != "Tie":
                                st.markdown(
                                    f'<div style="padding:10px;background:#f0fff0;border-radius:8px;border:1px solid #cce0cc;'
                                    f'text-align:center;">'
                                    f'<span style="font-size:14px;font-weight:800;color:{w_clr} !important;">'
                                    f'Winner: {winner}</span></div>', unsafe_allow_html=True)
                    else:
                        st.info("Need at least 2 sequences with 3+ occurrences to compare.")

        else:
            st.info("Not enough columns to determine pitch sequences.")

    # ─── Tab: Count Leverage ──────────────────────
    with tab_count:
        section_header("Count Leverage Analysis")
        st.caption("Optimal pitch selection by count — what works best in each situation")

        pdf_c = pdf.dropna(subset=["Balls", "Strikes"]).copy()
        pdf_c["Balls"] = pdf_c["Balls"].astype(int)
        pdf_c["Strikes"] = pdf_c["Strikes"].astype(int)
        pdf_c["Count"] = pdf_c["Balls"].astype(str) + "-" + pdf_c["Strikes"].astype(str)

        pitch_types = sorted(pdf["TaggedPitchType"].dropna().unique())

        # For each count, show best pitch type by whiff% and CSW%
        section_header("Best Pitch by Count (Whiff%)")
        count_pitch_data = {}
        for b in range(4):
            for s in range(3):
                count_str = f"{b}-{s}"
                cd = pdf_c[(pdf_c["Balls"] == b) & (pdf_c["Strikes"] == s)]
                if len(cd) < 5:
                    continue
                best_pt = None
                best_whiff = -1
                pt_results = {}
                for pt in pitch_types:
                    pt_cd = cd[cd["TaggedPitchType"] == pt]
                    if len(pt_cd) < 3:
                        continue
                    sw = pt_cd[pt_cd["PitchCall"].isin(SWING_CALLS)]
                    wh = pt_cd[pt_cd["PitchCall"] == "StrikeSwinging"]
                    csw = pt_cd[pt_cd["PitchCall"].isin(["StrikeSwinging", "StrikeCalled"])]
                    bt = pt_cd[pt_cd["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
                    whiff = len(wh) / max(len(sw), 1) * 100 if len(sw) > 0 else 0
                    csw_pct = len(csw) / len(pt_cd) * 100
                    ev = bt["ExitSpeed"].mean() if len(bt) > 0 else np.nan
                    pt_results[pt] = {"Usage": f"{len(pt_cd)/len(cd)*100:.0f}%", "Whiff%": whiff,
                                       "CSW%": csw_pct, "Avg EV": ev, "n": len(pt_cd)}
                    if whiff > best_whiff:
                        best_whiff = whiff
                        best_pt = pt
                count_pitch_data[count_str] = {"best": best_pt, "best_whiff": best_whiff, "details": pt_results, "total": len(cd)}

        # Display as grid
        grid_best = [['' for _ in range(3)] for _ in range(4)]
        grid_whiff = np.full((4, 3), np.nan)
        for b in range(4):
            for s in range(3):
                k = f"{b}-{s}"
                if k in count_pitch_data and count_pitch_data[k]["best"]:
                    d = count_pitch_data[k]
                    grid_best[b][s] = f"{d['best']}\n{d['best_whiff']:.0f}%"
                    grid_whiff[b][s] = d["best_whiff"]

        fig_best = go.Figure(data=go.Heatmap(
            z=grid_whiff, text=grid_best, texttemplate="%{text}",
            x=["0 Strikes", "1 Strike", "2 Strikes"], y=["0 Balls", "1 Ball", "2 Balls", "3 Balls"],
            colorscale=[[0, "#f7f7f7"], [0.5, "#f7c631"], [1, "#d22d49"]],
            zmin=0, zmax=50, showscale=True,
            colorbar=dict(title="Whiff%", len=0.8),
            textfont=dict(size=11),
        ))
        fig_best.update_layout(**CHART_LAYOUT, height=350, title="Best Pitch + Whiff% by Count")
        st.plotly_chart(fig_best, use_container_width=True)

        # Pitch usage by count
        section_header("Pitch Usage by Count")
        usage_rows = []
        for b in range(4):
            for s in range(3):
                cd = pdf_c[(pdf_c["Balls"] == b) & (pdf_c["Strikes"] == s)]
                if len(cd) < 5:
                    continue
                row = {"Count": f"{b}-{s}", "Total": len(cd)}
                for pt in pitch_types:
                    row[pt] = f"{len(cd[cd['TaggedPitchType'] == pt])/len(cd)*100:.0f}%"
                usage_rows.append(row)
        if usage_rows:
            st.dataframe(pd.DataFrame(usage_rows).set_index("Count"), use_container_width=True)

        # Detailed count breakdown
        section_header("Detailed Count Cheat Sheet")
        selected_count = st.selectbox("Select Count", [f"{b}-{s}" for b in range(4) for s in range(3)], key=f"{key_prefix}_count")
        if selected_count in count_pitch_data:
            details = count_pitch_data[selected_count]["details"]
            detail_rows = []
            for pt, d in sorted(details.items(), key=lambda x: -x[1]["Whiff%"]):
                detail_rows.append({
                    "Pitch Type": pt,
                    "Usage": d["Usage"],
                    "n": d["n"],
                    "Whiff%": f"{d['Whiff%']:.1f}%",
                    "CSW%": f"{d['CSW%']:.1f}%",
                    "Avg EV": f"{d['Avg EV']:.1f}" if not pd.isna(d["Avg EV"]) else "-",
                })
            if detail_rows:
                st.dataframe(pd.DataFrame(detail_rows).set_index("Pitch Type"), use_container_width=True)
                best = detail_rows[0]
                st.success(f"**Recommendation in {selected_count}:** Throw **{detail_rows[0]['Pitch Type']}** — "
                           f"{detail_rows[0]['Whiff%']} whiff rate, {detail_rows[0]['CSW%']} CSW%")

        # Ahead vs Behind vs Even
        section_header("Situational Pitch Selection")
        pdf_c["Situation"] = "Even"
        pdf_c.loc[pdf_c["Balls"] < pdf_c["Strikes"], "Situation"] = "Ahead"
        pdf_c.loc[pdf_c["Balls"] > pdf_c["Strikes"], "Situation"] = "Behind"

        sit_rows = []
        for sit in ["Ahead", "Even", "Behind"]:
            sit_df = pdf_c[pdf_c["Situation"] == sit]
            if len(sit_df) < 10:
                continue
            sw = sit_df[sit_df["PitchCall"].isin(SWING_CALLS)]
            wh = sit_df[sit_df["PitchCall"] == "StrikeSwinging"]
            csw = sit_df[sit_df["PitchCall"].isin(["StrikeSwinging", "StrikeCalled"])]
            bt = sit_df[sit_df["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            top_pt = sit_df["TaggedPitchType"].value_counts().index[0] if len(sit_df) > 0 else "-"
            sit_rows.append({
                "Situation": sit,
                "Pitches": len(sit_df),
                "Top Pitch": top_pt,
                "Zone%": f"{in_zone_mask(sit_df).sum()/len(sit_df[sit_df['PlateLocSide'].notna()])*100:.1f}%" if sit_df["PlateLocSide"].notna().any() else "-",
                "Whiff%": f"{len(wh)/len(sw)*100:.1f}%" if len(sw) > 0 else "-",
                "CSW%": f"{len(csw)/len(sit_df)*100:.1f}%" if len(sit_df) > 0 else "-",
                "Avg EV": f"{bt['ExitSpeed'].mean():.1f}" if len(bt) > 0 else "-",
            })
        if sit_rows:
            st.dataframe(pd.DataFrame(sit_rows).set_index("Situation"), use_container_width=True)

    # ─── Tab: Effective Velocity ──────────────────
    with tab_effv:
        section_header("Effective Velocity Analysis")
        st.caption("Perceived velocity based on pitch location and extension — a 91mph fastball up-and-in plays like 94+ mph")

        ev_df = pdf.dropna(subset=["RelSpeed", "PlateLocSide", "PlateLocHeight"]).copy()
        if len(ev_df) < 20:
            st.info("Not enough location data for effective velocity analysis.")
        else:
            # Compute effective velocity if not present
            if "EffectiveVelo" in ev_df.columns and ev_df["EffectiveVelo"].notna().sum() > len(ev_df) * 0.5:
                ev_df["EffVelo"] = ev_df["EffectiveVelo"]
            else:
                # Estimate: up-and-in adds ~2-3 mph, down-and-away subtracts ~2-3 mph
                # Hitter's reaction zone: pitches up and glove-side arrive "faster"
                loc_adj = (ev_df["PlateLocHeight"] - 2.5) * 1.5 + ev_df["PlateLocSide"].abs() * (-0.5)
                ev_df["EffVelo"] = ev_df["RelSpeed"] + loc_adj

            col_scatter, col_diff = st.columns(2)
            with col_scatter:
                section_header("Effective Velocity by Location")
                fig_effv = go.Figure()
                fig_effv.add_trace(go.Scatter(
                    x=ev_df["PlateLocSide"], y=ev_df["PlateLocHeight"],
                    mode="markers",
                    marker=dict(size=6, color=ev_df["EffVelo"],
                                colorscale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                                cmin=ev_df["EffVelo"].quantile(0.05),
                                cmax=ev_df["EffVelo"].quantile(0.95),
                                showscale=True, colorbar=dict(title="Eff Velo", len=0.8),
                                line=dict(width=0.3, color="white")),
                    hovertemplate="Actual: %{customdata[0]:.1f}<br>Effective: %{marker.color:.1f}<extra></extra>",
                    customdata=ev_df[["RelSpeed"]].values,
                    showlegend=False,
                ))
                add_strike_zone(fig_effv)
                fig_effv.update_layout(**CHART_LAYOUT, height=420,
                                        xaxis=dict(range=[-2.5, 2.5], title="Horizontal", scaleanchor="y"),
                                        yaxis=dict(range=[0, 5], title="Vertical"))
                st.plotly_chart(fig_effv, use_container_width=True)

            with col_diff:
                section_header("Velo Differential (Effective - Actual)")
                ev_df["VeloDiff"] = ev_df["EffVelo"] - ev_df["RelSpeed"]
                fig_diff = go.Figure()
                fig_diff.add_trace(go.Scatter(
                    x=ev_df["PlateLocSide"], y=ev_df["PlateLocHeight"],
                    mode="markers",
                    marker=dict(size=6, color=ev_df["VeloDiff"],
                                colorscale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                                cmid=0, showscale=True,
                                colorbar=dict(title="Diff", len=0.8),
                                line=dict(width=0.3, color="white")),
                    hovertemplate="Diff: %{marker.color:+.1f} mph<extra></extra>",
                    showlegend=False,
                ))
                add_strike_zone(fig_diff)
                fig_diff.update_layout(**CHART_LAYOUT, height=420,
                                        xaxis=dict(range=[-2.5, 2.5], title="Horizontal", scaleanchor="y"),
                                        yaxis=dict(range=[0, 5], title="Vertical"))
                st.plotly_chart(fig_diff, use_container_width=True)

            # Effective velo by pitch type
            section_header("Effective Velocity by Pitch Type")
            effv_pt_rows = []
            for pt in sorted(ev_df["TaggedPitchType"].dropna().unique()):
                pt_df = ev_df[ev_df["TaggedPitchType"] == pt]
                if len(pt_df) < 5:
                    continue
                effv_pt_rows.append({
                    "Pitch Type": pt,
                    "Actual Velo": f"{pt_df['RelSpeed'].mean():.1f}",
                    "Eff Velo": f"{pt_df['EffVelo'].mean():.1f}",
                    "Diff": f"{(pt_df['EffVelo'] - pt_df['RelSpeed']).mean():+.1f}",
                    "Max Eff": f"{pt_df['EffVelo'].max():.1f}",
                    "Min Eff": f"{pt_df['EffVelo'].min():.1f}",
                })
            if effv_pt_rows:
                st.dataframe(pd.DataFrame(effv_pt_rows).set_index("Pitch Type"), use_container_width=True)

            # Velo tunneling — show how pitch types overlap in effective velo
            section_header("Velocity Tunneling")
            st.caption("When different pitches arrive at similar effective velocities, hitters can't distinguish them")
            fig_tunnel = go.Figure()
            for pt in sorted(ev_df["TaggedPitchType"].dropna().unique()):
                pt_df = ev_df[ev_df["TaggedPitchType"] == pt]
                if len(pt_df) < 10:
                    continue
                clr = PITCH_COLORS.get(pt, "#aaa")
                fig_tunnel.add_trace(go.Violin(
                    y=pt_df["EffVelo"], name=pt,
                    box_visible=True, meanline_visible=True,
                    fillcolor=clr, line_color=clr, opacity=0.6,
                ))
            fig_tunnel.update_layout(**CHART_LAYOUT, height=380, showlegend=False,
                                      yaxis_title="Effective Velocity (mph)")
            st.plotly_chart(fig_tunnel, use_container_width=True)


# ──────────────────────────────────────────────
# DEFENSIVE POSITIONING
# ──────────────────────────────────────────────

def _compute_spray_zones(batted_df, batter_side):
    """Compute spray zone stats from batted ball data.
    Returns dict of zone → {centroid_x, centroid_y, count, out_rate, avg_ev, avg_dist, hit_rate}."""
    df = batted_df.dropna(subset=["Direction", "Distance"]).copy()
    if df.empty:
        return {}
    angle_rad = np.radians(df["Direction"])
    df["x"] = df["Distance"] * np.sin(angle_rad)
    df["y"] = df["Distance"] * np.cos(angle_rad)

    # Classify direction: Pull/Center/Oppo based on batter side
    # Trackman: positive Direction = right field
    if batter_side == "Right":
        df["FieldDir"] = np.where(df["Direction"] < -15, "Pull",
                         np.where(df["Direction"] > 15, "Oppo", "Center"))
    else:
        df["FieldDir"] = np.where(df["Direction"] > 15, "Pull",
                         np.where(df["Direction"] < -15, "Oppo", "Center"))

    df["FieldDepth"] = np.where(df["Distance"] < 180, "IF", "OF")
    df["Zone"] = df["FieldDepth"] + "-" + df["FieldDir"]

    zones = {}
    for zone_name, zdf in df.groupby("Zone"):
        is_out = zdf["PlayResult"].isin(["Out", "Sacrifice", "FieldersChoice"]) if "PlayResult" in zdf.columns else pd.Series([False]*len(zdf))
        is_hit = zdf["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]) if "PlayResult" in zdf.columns else pd.Series([False]*len(zdf))
        zones[zone_name] = {
            "centroid_x": zdf["x"].mean(),
            "centroid_y": zdf["y"].mean(),
            "count": len(zdf),
            "out_rate": is_out.sum() / max(len(zdf), 1) * 100,
            "hit_rate": is_hit.sum() / max(len(zdf), 1) * 100,
            "avg_ev": zdf["ExitSpeed"].mean() if "ExitSpeed" in zdf.columns and zdf["ExitSpeed"].notna().any() else np.nan,
            "avg_dist": zdf["Distance"].mean(),
        }
    return zones


def _recommend_fielder_positions(batted_df, batter_side):
    """Compute recommended fielder positions based on spray data.
    Returns dict of position_name → (x, y) in spray chart coordinates."""
    df = batted_df.dropna(subset=["Direction", "Distance"]).copy()
    if df.empty:
        return {}
    angle_rad = np.radians(df["Direction"])
    df["x"] = df["Distance"] * np.sin(angle_rad)
    df["y"] = df["Distance"] * np.cos(angle_rad)

    # Classify direction
    if batter_side == "Right":
        df["FieldDir"] = np.where(df["Direction"] < -15, "Pull",
                         np.where(df["Direction"] > 15, "Oppo", "Center"))
    else:
        df["FieldDir"] = np.where(df["Direction"] > 15, "Pull",
                         np.where(df["Direction"] < -15, "Oppo", "Center"))

    # Weight by damage potential: higher EV + hits weighted more
    df["weight"] = 1.0
    if "ExitSpeed" in df.columns:
        df.loc[df["ExitSpeed"].notna(), "weight"] = df.loc[df["ExitSpeed"].notna(), "ExitSpeed"] / 85.0
    if "PlayResult" in df.columns:
        df.loc[df["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]), "weight"] *= 1.5

    # Compute weighted centroids for each fielder position
    positions = {}
    # Ground balls → infielders
    gb = df[(df["TaggedHitType"] == "GroundBall") | (df["Distance"] < 180)] if "TaggedHitType" in df.columns else df[df["Distance"] < 180]
    # Air balls → outfielders
    air = df[(df["TaggedHitType"].isin(["FlyBall", "LineDrive"])) | (df["Distance"] >= 180)] if "TaggedHitType" in df.columns else df[df["Distance"] >= 180]

    pull_sign = -1 if batter_side == "Right" else 1

    # Infield positions — weighted centroids by pull/center/oppo
    gb_pull = gb[gb["FieldDir"] == "Pull"]
    gb_center = gb[gb["FieldDir"] == "Center"]
    gb_oppo = gb[gb["FieldDir"] == "Oppo"]

    # 3B: Pull side infield
    if batter_side == "Right":
        positions["3B"] = _weighted_centroid(gb_pull, fallback_x=-60, fallback_y=80)
        positions["1B"] = _weighted_centroid(gb_oppo, fallback_x=60, fallback_y=80)
    else:
        positions["3B"] = _weighted_centroid(gb_oppo, fallback_x=-60, fallback_y=80)
        positions["1B"] = _weighted_centroid(gb_pull, fallback_x=60, fallback_y=80)

    # SS and 2B: middle infield
    ss_df = gb[(gb["x"] < 0) & (gb["Distance"].between(50, 200))]
    _2b_df = gb[(gb["x"] >= 0) & (gb["Distance"].between(50, 200))]
    positions["SS"] = _weighted_centroid(ss_df, fallback_x=-40, fallback_y=120)
    positions["2B"] = _weighted_centroid(_2b_df, fallback_x=40, fallback_y=120)

    # Outfield positions
    of_pull = air[air["FieldDir"] == "Pull"]
    of_center = air[air["FieldDir"] == "Center"]
    of_oppo = air[air["FieldDir"] == "Oppo"]

    if batter_side == "Right":
        positions["LF"] = _weighted_centroid(of_pull, fallback_x=-180, fallback_y=220)
        positions["RF"] = _weighted_centroid(of_oppo, fallback_x=180, fallback_y=220)
    else:
        positions["LF"] = _weighted_centroid(of_oppo, fallback_x=-180, fallback_y=220)
        positions["RF"] = _weighted_centroid(of_pull, fallback_x=180, fallback_y=220)

    positions["CF"] = _weighted_centroid(of_center, fallback_x=0, fallback_y=280)

    return positions


def _weighted_centroid(df, fallback_x, fallback_y):
    """Compute damage-weighted centroid of batted balls."""
    if df.empty or len(df) < 2:
        return (fallback_x, fallback_y)
    w = df["weight"].values if "weight" in df.columns else np.ones(len(df))
    w_sum = w.sum()
    if w_sum == 0:
        return (fallback_x, fallback_y)
    cx = np.average(df["x"].values, weights=w)
    cy = np.average(df["y"].values, weights=w)
    return (cx, cy)


def _load_positioning_csvs(game_ids):
    """Load positioning CSVs matching given GameIDs.
    Scans DATA_ROOT parent and home directory for *_playerpositioning_FHC.csv files."""
    import re
    search_dirs = [
        os.path.dirname(DATA_ROOT),  # parent of v3
        os.path.expanduser("~"),     # home directory
        _APP_DIR,                    # app directory
    ]
    pos_files = []
    for d in search_dirs:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith("_playerpositioning_FHC.csv"):
                    pos_files.append(os.path.join(d, f))

    if not pos_files:
        return pd.DataFrame()

    frames = []
    for fp in pos_files:
        fname = os.path.basename(fp)
        # Extract GameID from filename: YYYYMMDD-Venue-Type-Num_unverified_playerpositioning_FHC.csv
        match = re.match(r"(.+?)_unverified_playerpositioning", fname)
        if not match:
            continue
        file_game_id = match.group(1)
        if game_ids is not None and file_game_id not in game_ids:
            continue
        try:
            df = pd.read_csv(fp)
            df["_GameID"] = file_game_id
            frames.append(df)
        except Exception:
            continue

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def _draw_defensive_field(batted_df, recommended=None, actual_positions=None, height=500,
                          color_by="PlayResult", title=""):
    """Draw baseball field with spray data and defensive positions overlaid."""
    fig = go.Figure()

    # Grass fill
    theta_grass = np.linspace(-np.pi / 4, np.pi / 4, 80)
    grass_r = 400
    grass_x = [0] + list(grass_r * np.sin(theta_grass)) + [0]
    grass_y = [0] + list(grass_r * np.cos(theta_grass)) + [0]
    fig.add_trace(go.Scatter(x=grass_x, y=grass_y, mode="lines",
                             fill="toself", fillcolor="rgba(76,160,60,0.06)",
                             line=dict(color="rgba(76,160,60,0.15)", width=1), showlegend=False,
                             hoverinfo="skip"))

    # Infield diamond
    diamond_x = [0, 63.6, 0, -63.6, 0]
    diamond_y = [0, 63.6, 127.3, 63.6, 0]
    fig.add_trace(go.Scatter(x=diamond_x, y=diamond_y, mode="lines",
                             line=dict(color="rgba(160,120,60,0.25)", width=1), showlegend=False,
                             fill="toself", fillcolor="rgba(160,120,60,0.06)", hoverinfo="skip"))

    # Foul lines
    fl = 350
    fig.add_trace(go.Scatter(x=[0, -fl * np.sin(np.pi / 4)], y=[0, fl * np.cos(np.pi / 4)],
                             mode="lines", line=dict(color="rgba(0,0,0,0.12)", width=1),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0, fl * np.sin(np.pi / 4)], y=[0, fl * np.cos(np.pi / 4)],
                             mode="lines", line=dict(color="rgba(0,0,0,0.12)", width=1),
                             showlegend=False, hoverinfo="skip"))

    # Batted balls
    if batted_df is not None and not batted_df.empty:
        spray = batted_df.dropna(subset=["Direction", "Distance"]).copy()
        if not spray.empty:
            angle_rad = np.radians(spray["Direction"])
            spray["x"] = spray["Distance"] * np.sin(angle_rad)
            spray["y"] = spray["Distance"] * np.cos(angle_rad)

            if color_by == "PlayResult" and "PlayResult" in spray.columns:
                result_colors = {"Out": "#999", "Single": "#2ca02c", "Double": "#f7c631",
                                 "Triple": "#fe6100", "HomeRun": "#d22d49",
                                 "Error": "#9467bd", "FieldersChoice": "#bbb",
                                 "Sacrifice": "#bbb"}
                for res, clr in result_colors.items():
                    sub = spray[spray["PlayResult"] == res]
                    if sub.empty:
                        continue
                    fig.add_trace(go.Scatter(
                        x=sub["x"], y=sub["y"], mode="markers",
                        marker=dict(size=6, color=clr, opacity=0.7,
                                    line=dict(width=0.3, color="white")),
                        name=res,
                        hovertemplate="EV: %{customdata[0]:.1f}<br>Dist: %{customdata[1]:.0f}ft<extra></extra>",
                        customdata=sub[["ExitSpeed", "Distance"]].fillna(0).values,
                    ))
            else:
                ev_vals = spray["ExitSpeed"].fillna(80) if "ExitSpeed" in spray.columns else pd.Series([80]*len(spray))
                fig.add_trace(go.Scatter(
                    x=spray["x"], y=spray["y"], mode="markers",
                    marker=dict(size=6, color=ev_vals,
                                colorscale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                                cmin=60, cmax=105, showscale=True,
                                colorbar=dict(title="EV", len=0.6),
                                line=dict(width=0.3, color="white")),
                    name="Batted Balls", showlegend=False,
                ))

    # Recommended positions (star markers)
    if recommended:
        pos_colors = {"1B": "#d62728", "2B": "#ff7f0e", "SS": "#2ca02c",
                      "3B": "#1f77b4", "LF": "#9467bd", "CF": "#8c564b", "RF": "#e377c2"}
        for pos_name, (px, py) in recommended.items():
            fig.add_trace(go.Scatter(
                x=[px], y=[py], mode="markers+text",
                marker=dict(size=18, color=pos_colors.get(pos_name, "#333"),
                            symbol="star", line=dict(width=2, color="white")),
                text=[pos_name], textposition="top center",
                textfont=dict(size=11, color=pos_colors.get(pos_name, "#333")),
                name=f"Rec: {pos_name}", showlegend=False,
                hovertemplate=f"<b>{pos_name}</b><br>Recommended<br>({px:.0f}, {py:.0f})<extra></extra>",
            ))

    # Actual positions (diamond markers)
    if actual_positions:
        for pos_name, (px, py) in actual_positions.items():
            fig.add_trace(go.Scatter(
                x=[px], y=[py], mode="markers+text",
                marker=dict(size=14, color="#1a1a2e", symbol="diamond",
                            line=dict(width=2, color="white")),
                text=[pos_name], textposition="bottom center",
                textfont=dict(size=9, color="#1a1a2e"),
                name=f"Actual: {pos_name}", showlegend=False,
                hovertemplate=f"<b>{pos_name}</b><br>Actual Position<br>({px:.0f}, {py:.0f})<extra></extra>",
            ))

    fig.update_layout(
        xaxis=dict(range=[-300, 300], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-15, 400], showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", fixedrange=True),
        height=height, margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="#1a1a2e", family="Inter, Arial, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5,
                    font=dict(size=10, color="#1a1a2e"), bgcolor="rgba(0,0,0,0)"),
        title=dict(text=title, font=dict(size=14)),
    )
    return fig


def page_defensive_positioning(data):
    st.markdown('<div class="section-header">Defensive Positioning</div>', unsafe_allow_html=True)
    st.caption("Spray-based defensive alignment optimizer — see where batted balls land and where fielders should stand")

    # Mode selection
    mode = st.radio("Analysis Mode", ["Scout Batter", "Team Tendencies"], horizontal=True, key="dp_mode")

    if mode == "Scout Batter":
        # All batters in the database (via DuckDB)
        _bp = query_population("SELECT DISTINCT Batter FROM trackman WHERE PitchCall='InPlay' AND Batter IS NOT NULL ORDER BY Batter")
        batters = _bp["Batter"].tolist()
        if not batters:
            st.warning("No batted ball data found.")
            return
        col1, col2 = st.columns([2, 1])
        with col1:
            batter = st.selectbox("Select Batter", batters, format_func=display_name, key="dp_batter")
        with col2:
            seasons = get_all_seasons()
            season_filter = st.multiselect("Season", seasons, default=seasons, key="dp_season")
        _season_clause = f"AND Season IN ({','.join(str(int(s)) for s in season_filter)})" if season_filter else ""
        bdf = query_population(f"SELECT * FROM trackman WHERE Batter = '{batter.replace(chr(39), chr(39)+chr(39))}' {_season_clause}")
        label = display_name(batter)
    else:
        _tp = query_population("SELECT DISTINCT BatterTeam FROM trackman WHERE BatterTeam IS NOT NULL ORDER BY BatterTeam")
        teams = _tp["BatterTeam"].tolist()
        col1, col2 = st.columns([2, 1])
        with col1:
            team = st.selectbox("Select Team", teams, key="dp_team")
        with col2:
            seasons = get_all_seasons()
            season_filter = st.multiselect("Season", seasons, default=seasons, key="dp_season_t")
        _season_clause = f"AND Season IN ({','.join(str(int(s)) for s in season_filter)})" if season_filter else ""
        bdf = query_population(f"SELECT * FROM trackman WHERE BatterTeam = '{team}' {_season_clause}")
        label = team

    batted = bdf[bdf["PitchCall"] == "InPlay"].copy()
    if len(batted) < 10:
        st.warning(f"Not enough batted balls for {label} (need 10+, have {len(batted)}).")
        return

    # Determine batter side
    batter_side = "Right"
    if "BatterSide" in bdf.columns and bdf["BatterSide"].notna().any():
        side_mode = bdf["BatterSide"].mode()
        if len(side_mode) > 0:
            batter_side = side_mode.iloc[0]
    b_str = {"Right": "R", "Left": "L"}.get(batter_side, batter_side)

    st.markdown(
        f'<div style="background:#f8f8f8;padding:10px 16px;border-radius:8px;margin:8px 0;border:1px solid #eee;">'
        f'<span style="font-size:16px;font-weight:800;color:#1a1a2e !important;">{label}</span>'
        f'<span style="margin-left:12px;font-size:13px;color:#666 !important;">Bats: {b_str} | '
        f'{len(batted)} batted balls</span></div>', unsafe_allow_html=True)

    tab_spray, tab_shift, tab_sit, tab_actual = st.tabs([
        "Spray & Positioning", "Shift Analysis", "Situational", "Actual Positioning"
    ])

    # ─── Tab 1: Spray Tendencies & Optimal Positioning ──────
    with tab_spray:
        section_header("Spray Tendencies & Optimal Positioning")

        # Filters
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            ht_options = ["All"] + sorted(batted["TaggedHitType"].dropna().unique().tolist()) if "TaggedHitType" in batted.columns else ["All"]
            hit_type_filter = st.selectbox("Hit Type", ht_options, key="dp_ht")
        with col_f2:
            pt_options = ["All"] + sorted(batted["TaggedPitchType"].dropna().unique().tolist()) if "TaggedPitchType" in batted.columns else ["All"]
            pitch_type_filter = st.selectbox("Pitch Type Faced", pt_options, key="dp_pt")

        filt = batted.copy()
        if hit_type_filter != "All" and "TaggedHitType" in filt.columns:
            filt = filt[filt["TaggedHitType"] == hit_type_filter]
        if pitch_type_filter != "All" and "TaggedPitchType" in filt.columns:
            filt = filt[filt["TaggedPitchType"] == pitch_type_filter]

        if len(filt) < 5:
            st.info("Not enough batted balls with current filters.")
        else:
            # Compute recommended positions
            recommended = _recommend_fielder_positions(filt, batter_side)

            col_field, col_stats = st.columns([3, 2])
            with col_field:
                fig_field = _draw_defensive_field(filt, recommended=recommended, height=520,
                                                  title="Spray Chart + Recommended Fielder Positions")
                st.plotly_chart(fig_field, use_container_width=True)
                st.caption("★ Stars = recommended fielder positions (weighted by hit frequency × damage)")

            with col_stats:
                # Zone breakdown table
                section_header("Zone Breakdown")
                zones = _compute_spray_zones(filt, batter_side)
                zone_rows = []
                for zname in ["IF-Pull", "IF-Center", "IF-Oppo", "OF-Pull", "OF-Center", "OF-Oppo"]:
                    z = zones.get(zname, {})
                    if not z:
                        continue
                    zone_rows.append({
                        "Zone": zname,
                        "Count": z["count"],
                        "Hit%": f"{z['count']/len(filt)*100:.1f}%" if len(filt) > 0 else "-",
                        "Out%": f"{z['out_rate']:.1f}%",
                        "H Rate": f"{z['hit_rate']:.1f}%",
                        "Avg EV": f"{z['avg_ev']:.1f}" if not pd.isna(z.get("avg_ev", np.nan)) else "-",
                        "Avg Dist": f"{z['avg_dist']:.0f} ft",
                    })
                if zone_rows:
                    st.dataframe(pd.DataFrame(zone_rows).set_index("Zone"), use_container_width=True)

                # Pull/Center/Oppo summary
                section_header("Directional Summary")
                spray_df = filt.dropna(subset=["Direction"]).copy()
                if not spray_df.empty:
                    if batter_side == "Right":
                        spray_df["Dir"] = np.where(spray_df["Direction"] < -15, "Pull",
                                          np.where(spray_df["Direction"] > 15, "Oppo", "Center"))
                    else:
                        spray_df["Dir"] = np.where(spray_df["Direction"] > 15, "Pull",
                                          np.where(spray_df["Direction"] < -15, "Oppo", "Center"))

                    dir_rows = []
                    for d in ["Pull", "Center", "Oppo"]:
                        dd = spray_df[spray_df["Dir"] == d]
                        if dd.empty:
                            continue
                        gb_pct = len(dd[dd["TaggedHitType"] == "GroundBall"]) / max(len(dd), 1) * 100 if "TaggedHitType" in dd.columns else np.nan
                        ld_pct = len(dd[dd["TaggedHitType"] == "LineDrive"]) / max(len(dd), 1) * 100 if "TaggedHitType" in dd.columns else np.nan
                        fb_pct = len(dd[dd["TaggedHitType"] == "FlyBall"]) / max(len(dd), 1) * 100 if "TaggedHitType" in dd.columns else np.nan
                        dir_rows.append({
                            "Direction": d,
                            "BBE": len(dd),
                            "%": f"{len(dd)/len(spray_df)*100:.1f}%",
                            "Avg EV": f"{dd['ExitSpeed'].mean():.1f}" if dd["ExitSpeed"].notna().any() else "-",
                            "GB%": f"{gb_pct:.0f}%" if not pd.isna(gb_pct) else "-",
                            "LD%": f"{ld_pct:.0f}%" if not pd.isna(ld_pct) else "-",
                            "FB%": f"{fb_pct:.0f}%" if not pd.isna(fb_pct) else "-",
                        })
                    if dir_rows:
                        st.dataframe(pd.DataFrame(dir_rows).set_index("Direction"), use_container_width=True)

    # ─── Tab 2: Shift Analysis ──────────────────────
    with tab_shift:
        section_header("Shift Analysis & Recommendations")

        spray_all = batted.dropna(subset=["Direction"]).copy()
        if len(spray_all) < 10:
            st.info("Not enough batted balls for shift analysis.")
        else:
            if batter_side == "Right":
                spray_all["Dir"] = np.where(spray_all["Direction"] < -15, "Pull",
                                   np.where(spray_all["Direction"] > 15, "Oppo", "Center"))
            else:
                spray_all["Dir"] = np.where(spray_all["Direction"] > 15, "Pull",
                                   np.where(spray_all["Direction"] < -15, "Oppo", "Center"))

            total = len(spray_all)
            pull_pct = len(spray_all[spray_all["Dir"] == "Pull"]) / total * 100
            center_pct = len(spray_all[spray_all["Dir"] == "Center"]) / total * 100
            oppo_pct = len(spray_all[spray_all["Dir"] == "Oppo"]) / total * 100
            gb_pct = len(spray_all[spray_all["TaggedHitType"] == "GroundBall"]) / total * 100 if "TaggedHitType" in spray_all.columns else 0
            gb_pull = spray_all[(spray_all["TaggedHitType"] == "GroundBall") & (spray_all["Dir"] == "Pull")] if "TaggedHitType" in spray_all.columns else pd.DataFrame()
            n_gb = len(spray_all[spray_all["TaggedHitType"] == "GroundBall"]) if "TaggedHitType" in spray_all.columns else 0
            gb_pull_pct = len(gb_pull) / n_gb * 100 if n_gb > 0 else 0

            # Shift recommendation
            if pull_pct > 45 and gb_pct > 45:
                shift_rec = "Infield Shift"
                shift_color = "#d22d49"
                shift_desc = (f"Pull-heavy hitter ({pull_pct:.0f}% pull) with high GB rate ({gb_pct:.0f}%). "
                              f"Ground balls go pull-side {gb_pull_pct:.0f}% of the time. "
                              f"Shift infield toward pull side.")
            elif pull_pct > 40:
                shift_rec = "Shade Pull"
                shift_color = "#fe6100"
                shift_desc = (f"Moderate pull tendency ({pull_pct:.0f}%). "
                              f"Shade middle infielders slightly toward pull side, don't full shift.")
            elif oppo_pct > 40:
                shift_rec = "Shade Oppo"
                shift_color = "#1f77b4"
                shift_desc = (f"Oppo-oriented hitter ({oppo_pct:.0f}% opposite field). "
                              f"Shade defense toward opposite field.")
            else:
                shift_rec = "Standard"
                shift_color = "#2ca02c"
                shift_desc = (f"Balanced spray (Pull: {pull_pct:.0f}%, Center: {center_pct:.0f}%, Oppo: {oppo_pct:.0f}%). "
                              f"Use standard defensive alignment.")

            st.markdown(
                f'<div style="padding:16px;background:white;border-radius:10px;border-left:6px solid {shift_color};'
                f'border:1px solid #eee;margin:8px 0;">'
                f'<div style="font-size:20px;font-weight:900;color:{shift_color} !important;">{shift_rec}</div>'
                f'<div style="font-size:13px;color:#333 !important;margin-top:4px;">{shift_desc}</div>'
                f'</div>', unsafe_allow_html=True)

            # Spray distribution cards
            col_p, col_c, col_o = st.columns(3)
            for col, dir_name, pct in [(col_p, "Pull", pull_pct), (col_c, "Center", center_pct), (col_o, "Oppo", oppo_pct)]:
                with col:
                    dir_df = spray_all[spray_all["Dir"] == dir_name]
                    hits = dir_df[dir_df["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"])] if "PlayResult" in dir_df.columns else pd.DataFrame()
                    outs = dir_df[dir_df["PlayResult"].isin(["Out", "Sacrifice", "FieldersChoice"])] if "PlayResult" in dir_df.columns else pd.DataFrame()
                    ev_str = f"{dir_df['ExitSpeed'].mean():.1f}" if dir_df["ExitSpeed"].notna().any() else "-"
                    st.metric(dir_name, f"{pct:.1f}%",
                              delta=f"EV: {ev_str} | Hits: {len(hits)} | Outs: {len(outs)}")

            # Hit outcome by direction
            section_header("Hit Outcome by Field Third")
            outcome_rows = []
            for d in ["Pull", "Center", "Oppo"]:
                dd = spray_all[spray_all["Dir"] == d]
                if dd.empty or "PlayResult" not in dd.columns:
                    continue
                n = len(dd)
                outs = dd["PlayResult"].isin(["Out", "Sacrifice", "FieldersChoice"]).sum()
                singles = (dd["PlayResult"] == "Single").sum()
                xbh = dd["PlayResult"].isin(["Double", "Triple", "HomeRun"]).sum()
                outcome_rows.append({
                    "Direction": d,
                    "BBE": n,
                    "Out%": f"{outs/n*100:.1f}%",
                    "Single%": f"{singles/n*100:.1f}%",
                    "XBH%": f"{xbh/n*100:.1f}%",
                    "Avg EV": f"{dd['ExitSpeed'].mean():.1f}" if dd["ExitSpeed"].notna().any() else "-",
                    "Avg Dist": f"{dd['Distance'].mean():.0f}ft" if dd["Distance"].notna().any() else "-",
                })
            if outcome_rows:
                st.dataframe(pd.DataFrame(outcome_rows).set_index("Direction"), use_container_width=True)

            # GB vs FB directional tendencies
            if "TaggedHitType" in spray_all.columns:
                section_header("Ground Ball vs Fly Ball Directional Tendencies")
                st.caption("Ground balls are typically more pull-heavy — the shift exploits this")
                ht_dir_rows = []
                for ht in ["GroundBall", "LineDrive", "FlyBall"]:
                    ht_df = spray_all[spray_all["TaggedHitType"] == ht]
                    if len(ht_df) < 3:
                        continue
                    ht_n = len(ht_df)
                    ht_dir_rows.append({
                        "Hit Type": {"GroundBall": "Ground Ball", "LineDrive": "Line Drive", "FlyBall": "Fly Ball"}.get(ht, ht),
                        "Count": ht_n,
                        "Pull%": f"{len(ht_df[ht_df['Dir']=='Pull'])/ht_n*100:.1f}%",
                        "Center%": f"{len(ht_df[ht_df['Dir']=='Center'])/ht_n*100:.1f}%",
                        "Oppo%": f"{len(ht_df[ht_df['Dir']=='Oppo'])/ht_n*100:.1f}%",
                        "Avg EV": f"{ht_df['ExitSpeed'].mean():.1f}" if ht_df["ExitSpeed"].notna().any() else "-",
                    })
                if ht_dir_rows:
                    st.dataframe(pd.DataFrame(ht_dir_rows).set_index("Hit Type"), use_container_width=True)

                # Stacked bar chart: direction by hit type
                fig_stack = go.Figure()
                for d, clr in [("Pull", "#d22d49"), ("Center", "#f7c631"), ("Oppo", "#1f77b4")]:
                    vals = []
                    cats = []
                    for ht in ["GroundBall", "LineDrive", "FlyBall"]:
                        ht_df = spray_all[spray_all["TaggedHitType"] == ht]
                        if len(ht_df) < 3:
                            continue
                        vals.append(len(ht_df[ht_df["Dir"] == d]) / len(ht_df) * 100)
                        cats.append({"GroundBall": "GB", "LineDrive": "LD", "FlyBall": "FB"}.get(ht, ht))
                    fig_stack.add_trace(go.Bar(x=cats, y=vals, name=d, marker_color=clr))
                fig_stack.update_layout(**CHART_LAYOUT, height=300, barmode="group",
                                        yaxis_title="% of Hit Type", legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_stack, use_container_width=True)

    # ─── Tab 3: Situational Positioning ──────────────
    with tab_sit:
        section_header("Situational Positioning Adjustments")
        st.caption("How spray tendencies change by count, pitch type, and situation")

        spray_sit = batted.dropna(subset=["Direction"]).copy()
        if len(spray_sit) < 15:
            st.info("Not enough data for situational analysis.")
        else:
            if batter_side == "Right":
                spray_sit["Dir"] = np.where(spray_sit["Direction"] < -15, "Pull",
                                   np.where(spray_sit["Direction"] > 15, "Oppo", "Center"))
            else:
                spray_sit["Dir"] = np.where(spray_sit["Direction"] > 15, "Pull",
                                   np.where(spray_sit["Direction"] < -15, "Oppo", "Center"))

            # By count situation
            section_header("Spray by Count Situation")
            if "Balls" in spray_sit.columns and "Strikes" in spray_sit.columns:
                spray_sit_c = spray_sit.dropna(subset=["Balls", "Strikes"]).copy()
                spray_sit_c["Balls"] = spray_sit_c["Balls"].astype(int)
                spray_sit_c["Strikes"] = spray_sit_c["Strikes"].astype(int)
                spray_sit_c["Situation"] = "Even"
                spray_sit_c.loc[spray_sit_c["Balls"] < spray_sit_c["Strikes"], "Situation"] = "Pitcher Ahead"
                spray_sit_c.loc[spray_sit_c["Balls"] > spray_sit_c["Strikes"], "Situation"] = "Hitter Ahead"

                sit_rows = []
                for sit in ["Pitcher Ahead", "Even", "Hitter Ahead"]:
                    sdf = spray_sit_c[spray_sit_c["Situation"] == sit]
                    if len(sdf) < 5:
                        continue
                    n = len(sdf)
                    sit_rows.append({
                        "Situation": sit,
                        "BBE": n,
                        "Pull%": f"{len(sdf[sdf['Dir']=='Pull'])/n*100:.1f}%",
                        "Center%": f"{len(sdf[sdf['Dir']=='Center'])/n*100:.1f}%",
                        "Oppo%": f"{len(sdf[sdf['Dir']=='Oppo'])/n*100:.1f}%",
                        "Avg EV": f"{sdf['ExitSpeed'].mean():.1f}" if sdf["ExitSpeed"].notna().any() else "-",
                        "GB%": f"{len(sdf[sdf['TaggedHitType']=='GroundBall'])/n*100:.1f}%" if "TaggedHitType" in sdf.columns else "-",
                    })
                if sit_rows:
                    st.dataframe(pd.DataFrame(sit_rows).set_index("Situation"), use_container_width=True)

            # 2-strike approach
            section_header("2-Strike Approach")
            if "Strikes" in spray_sit.columns:
                pre2k = spray_sit[spray_sit["Strikes"].fillna(0).astype(int) < 2]
                with2k = spray_sit[spray_sit["Strikes"].fillna(0).astype(int) == 2]
                two_k_rows = []
                for label_2k, df_2k in [("< 2 Strikes", pre2k), ("2 Strikes", with2k)]:
                    if len(df_2k) < 5:
                        continue
                    n = len(df_2k)
                    two_k_rows.append({
                        "Count": label_2k,
                        "BBE": n,
                        "Pull%": f"{len(df_2k[df_2k['Dir']=='Pull'])/n*100:.1f}%",
                        "Center%": f"{len(df_2k[df_2k['Dir']=='Center'])/n*100:.1f}%",
                        "Oppo%": f"{len(df_2k[df_2k['Dir']=='Oppo'])/n*100:.1f}%",
                        "Avg EV": f"{df_2k['ExitSpeed'].mean():.1f}" if df_2k["ExitSpeed"].notna().any() else "-",
                        "GB%": f"{len(df_2k[df_2k['TaggedHitType']=='GroundBall'])/n*100:.1f}%" if "TaggedHitType" in df_2k.columns else "-",
                    })
                if two_k_rows:
                    st.dataframe(pd.DataFrame(two_k_rows).set_index("Count"), use_container_width=True)
                    # Insight
                    if len(pre2k) >= 5 and len(with2k) >= 5:
                        pre_pull = len(pre2k[pre2k["Dir"] == "Pull"]) / len(pre2k) * 100
                        with_pull = len(with2k[with2k["Dir"] == "Pull"]) / len(with2k) * 100
                        diff = with_pull - pre_pull
                        if abs(diff) > 5:
                            direction = "more oppo" if diff < 0 else "more pull-heavy"
                            st.info(f"With 2 strikes, this batter goes **{direction}** ({diff:+.1f}% pull change). "
                                    f"Adjust positioning accordingly.")

            # By pitch type faced
            section_header("Spray by Pitch Type Faced")
            if "TaggedPitchType" in spray_sit.columns:
                pt_types = spray_sit["TaggedPitchType"].value_counts()
                pt_types = pt_types[pt_types >= 5].index.tolist()
                if pt_types:
                    n_cols = min(len(pt_types), 3)
                    pt_cols = st.columns(n_cols)
                    for idx, pt in enumerate(pt_types[:6]):
                        pt_df = spray_sit[spray_sit["TaggedPitchType"] == pt]
                        with pt_cols[idx % n_cols]:
                            fig_pt = _draw_defensive_field(pt_df, height=300, title=pt, color_by="EV")
                            st.plotly_chart(fig_pt, use_container_width=True)
                            n = len(pt_df)
                            pull_p = len(pt_df[pt_df["Dir"] == "Pull"]) / max(n, 1) * 100
                            st.caption(f"n={n} | Pull: {pull_p:.0f}% | "
                                       f"EV: {pt_df['ExitSpeed'].mean():.1f}" if pt_df["ExitSpeed"].notna().any() else f"n={n} | Pull: {pull_p:.0f}%")

    # ─── Tab 4: Actual Positioning Data ──────────────
    with tab_actual:
        section_header("Actual Fielder Positioning (Camera Data)")
        st.caption("When field-home-camera positioning data is available, see where fielders were actually standing")

        # Try to load positioning files
        game_ids = bdf["GameID"].dropna().unique().tolist() if "GameID" in bdf.columns else []
        pos_data = _load_positioning_csvs(game_ids if game_ids else None)

        if pos_data.empty:
            st.info("No positioning CSV data found for the selected games. "
                    "Positioning data requires *_playerpositioning_FHC.csv files matching the game IDs.")
            st.markdown(
                '<div style="padding:12px;background:#f8f8f8;border-radius:8px;border:1px solid #eee;margin-top:8px;">'
                '<div style="font-size:13px;font-weight:700;color:#1a1a2e !important;">How Positioning Data Works</div>'
                '<div style="font-size:12px;color:#555 !important;">Trackman\'s field-home camera captures all 7 fielder '
                'positions (1B, 2B, 3B, SS, LF, CF, RF) at pitch release. When CSV files are available, this tab shows '
                'actual fielder positions overlaid on the batter\'s spray chart, along with shift detection and gap analysis.</div>'
                '</div>', unsafe_allow_html=True)
        else:
            st.success(f"Loaded positioning data: {len(pos_data)} pitches from {pos_data['_GameID'].nunique()} game(s)")

            # Convert positioning coordinates to spray chart coordinate system
            # Positioning: X = horizontal (1B side positive), Z = depth (3B side negative)
            # Spray chart: x = horizontal (right field positive), y = distance from home plate
            # The positioning uses a different coordinate system — X is depth, Z is lateral
            # Based on the data: 1B X~90,Z~72; 3B X~73,Z~-50; CF X~332,Z~2
            # This looks like X = distance from home, Z = lateral (positive = 1B side)
            # We need to map to spray chart: x = lateral, y = distance
            pos_cols = ["1B", "2B", "3B", "SS", "LF", "CF", "RF"]
            has_pos = all(f"{p}_PositionAtReleaseX" in pos_data.columns for p in pos_cols)

            if has_pos:
                # Show per-pitch positioning for InPlay pitches
                inplay_pos = pos_data[pos_data["PitchCall"] == "InPlay"].copy() if "PitchCall" in pos_data.columns else pos_data.copy()

                if len(inplay_pos) > 0:
                    # Compute average positions across all pitches
                    avg_positions = {}
                    for p in pos_cols:
                        x_col = f"{p}_PositionAtReleaseX"
                        z_col = f"{p}_PositionAtReleaseZ"
                        if pos_data[x_col].notna().any():
                            # Map: positioning Z → spray chart x (lateral), positioning X → spray chart y (depth)
                            avg_x = pos_data[z_col].mean()  # lateral position
                            avg_y = pos_data[x_col].mean()  # depth from home
                            avg_positions[p] = (avg_x, avg_y)

                    # Draw field with actual positions + spray data + recommended
                    recommended = _recommend_fielder_positions(batted, batter_side)
                    fig_actual = _draw_defensive_field(batted, recommended=recommended,
                                                       actual_positions=avg_positions,
                                                       height=550, title="Actual vs Recommended Positioning")
                    st.plotly_chart(fig_actual, use_container_width=True)
                    st.caption("★ Stars = recommended positions | ◆ Diamonds = actual average positions")

                    # Gap analysis table
                    if avg_positions and recommended:
                        section_header("Position Gap Analysis")
                        gap_rows = []
                        for p in pos_cols:
                            if p in avg_positions and p in recommended:
                                ax, ay = avg_positions[p]
                                rx, ry = recommended[p]
                                gap = np.sqrt((ax - rx)**2 + (ay - ry)**2)
                                gap_rows.append({
                                    "Position": p,
                                    "Actual (x, y)": f"({ax:.0f}, {ay:.0f})",
                                    "Recommended (x, y)": f"({rx:.0f}, {ry:.0f})",
                                    "Gap (ft)": f"{gap:.1f}",
                                    "Direction": "Shift pull-ward" if (rx - ax) * (-1 if batter_side == "Right" else 1) > 5
                                                 else "Shift oppo-ward" if (rx - ax) * (-1 if batter_side == "Right" else 1) < -5
                                                 else "Well-positioned",
                                })
                        if gap_rows:
                            st.dataframe(pd.DataFrame(gap_rows).set_index("Position"), use_container_width=True)

                # Shift detection summary
                if "DetectedShift" in pos_data.columns:
                    section_header("Detected Shift Usage")
                    shift_counts = pos_data["DetectedShift"].value_counts()
                    shift_rows = []
                    for shift_type, count in shift_counts.items():
                        shift_rows.append({
                            "Shift Type": shift_type,
                            "Pitches": count,
                            "Usage%": f"{count/len(pos_data)*100:.1f}%",
                        })
                    st.dataframe(pd.DataFrame(shift_rows).set_index("Shift Type"), use_container_width=True)
            else:
                st.warning("Positioning coordinate columns not found in the CSV data.")


# ──────────────────────────────────────────────
# PAGE: POSTGAME REPORT
# ──────────────────────────────────────────────
BALL_RADIUS = 0.12  # ~1.45 in ≈ baseball radius in feet

def _pg_slug(name):
    """Create a key-safe slug from a player name."""
    return name.replace(" ", "").replace(",", "").replace(".", "")


def _pg_estimate_ip(pdf):
    """Estimate innings pitched from pitch-level data."""
    outs = 0
    if "OutsOnPlay" in pdf.columns:
        outs += pd.to_numeric(pdf["OutsOnPlay"], errors="coerce").fillna(0).sum()
    if "KorBB" in pdf.columns:
        outs += len(pdf[pdf["KorBB"] == "Strikeout"])
        outs -= len(pdf[(pdf["KorBB"] == "Strikeout") & (pdf.get("OutsOnPlay", pd.Series(dtype=float)).fillna(0) > 0)]) if "OutsOnPlay" in pdf.columns else 0
    full = int(outs // 3)
    part = int(outs % 3)
    return f"{full}.{part}"


def _pg_count_state(balls, strikes):
    """Classify count from pitcher's perspective: Ahead, Behind, Even."""
    if pd.isna(balls) or pd.isna(strikes):
        return "Even"
    b, s = int(balls), int(strikes)
    if s > b:
        return "Ahead"
    elif b > s:
        return "Behind"
    return "Even"


def _pg_inning_group(inning):
    """Classify inning into Early/Mid/Late."""
    if pd.isna(inning):
        return "Early"
    i = int(inning)
    if i <= 3:
        return "Early (1-3)"
    elif i <= 6:
        return "Mid (4-6)"
    return "Late (7+)"


def _pg_count_leverage(balls, strikes):
    """Classify count leverage: High if 3-ball or 2-strike count, else Medium/Low."""
    if pd.isna(balls) or pd.isna(strikes):
        return "Low"
    b, s = int(balls), int(strikes)
    if b == 3 or s == 2:
        return "High"
    if b == 2 or s == 1:
        return "Medium"
    return "Low"


def _pg_pitch_sequence_text(ab_df):
    """Build a compact text description of a pitch sequence for an at-bat."""
    parts = []
    for _, row in ab_df.iterrows():
        pt = row.get("TaggedPitchType", "?")
        velo = row.get("RelSpeed", np.nan)
        call = row.get("PitchCall", "?")
        v_str = f" {velo:.0f}" if pd.notna(velo) else ""
        call_short = {"StrikeCalled": "SC", "BallCalled": "BC", "StrikeSwinging": "SS",
                      "FoulBall": "F", "FoulBallNotFieldable": "F", "FoulBallFieldable": "F",
                      "InPlay": "IP", "HitByPitch": "HBP", "BallIntentional": "IB"}.get(call, call[:3] if isinstance(call, str) else "?")
        parts.append(f"{pt}{v_str} ({call_short})")
    return " → ".join(parts)


def _pg_mini_location_plot(ab_df, key_suffix=""):
    """Create a small location scatter for a single at-bat with numbered pitches."""
    loc = ab_df.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
    if loc.empty:
        return None
    loc = loc.reset_index(drop=True)
    loc["PitchNum"] = range(1, len(loc) + 1)
    fig = go.Figure()
    for _, row in loc.iterrows():
        pt = row.get("TaggedPitchType", "Other")
        color = PITCH_COLORS.get(pt, "#aaa")
        fig.add_trace(go.Scatter(
            x=[row["PlateLocSide"]], y=[row["PlateLocHeight"]],
            mode="markers+text", text=[str(int(row["PitchNum"]))],
            textposition="top center", textfont=dict(size=9, color="#1a1a2e"),
            marker=dict(size=10, color=color, line=dict(width=1, color="white")),
            showlegend=False,
            hovertemplate=f"#{int(row['PitchNum'])} {pt}<br>{row.get('PitchCall','')}<extra></extra>",
        ))
    add_strike_zone(fig)
    fig.update_layout(
        xaxis=dict(range=[-2.5, 2.5], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, scaleanchor="y"),
        yaxis=dict(range=[0, 5], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        height=220, margin=dict(l=5, r=5, t=5, b=5),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


# ── Umpire Report ──
def _postgame_umpire(gd):
    """Render the Umpire Report tab for a single game."""
    section_header("Umpire Report")

    called = gd[gd["PitchCall"].isin(["StrikeCalled", "BallCalled"])].copy()
    called = called.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    if called.empty:
        st.info("No called pitch location data available for this game.")
        return

    # Fixed rulebook zone for umpire evaluation (no batter-adaptive)
    iz = (called["PlateLocSide"].abs() <= ZONE_SIDE) & \
         called["PlateLocHeight"].between(ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP)
    is_strike = called["PitchCall"] == "StrikeCalled"
    called["InZone"] = iz
    called["Correct"] = (is_strike & iz) | (~is_strike & ~iz)
    called["Gifted"] = is_strike & ~iz
    called["Missed"] = ~is_strike & iz

    # ── 3b/3c: Accuracy scatter + metrics ──
    col_scatter, col_metrics = st.columns([2, 1])

    with col_scatter:
        section_header("Called Pitch Accuracy")
        correct = called[called["Correct"]]
        incorrect = called[~called["Correct"]]
        fig = go.Figure()
        if not correct.empty:
            fig.add_trace(go.Scatter(
                x=correct["PlateLocSide"], y=correct["PlateLocHeight"],
                mode="markers", marker=dict(size=7, color="#2ca02c", symbol="circle", opacity=0.7),
                name="Correct",
                customdata=correct[["PitchCall", "Batter", "Pitcher", "Inning"]].fillna("").values if all(c in correct.columns for c in ["PitchCall","Batter","Pitcher","Inning"]) else None,
                hovertemplate="%{customdata[0]}<br>Batter: %{customdata[1]}<br>Pitcher: %{customdata[2]}<br>Inn: %{customdata[3]}<extra></extra>" if all(c in correct.columns for c in ["PitchCall","Batter","Pitcher","Inning"]) else None,
            ))
        if not incorrect.empty:
            fig.add_trace(go.Scatter(
                x=incorrect["PlateLocSide"], y=incorrect["PlateLocHeight"],
                mode="markers", marker=dict(size=9, color="#d62728", symbol="x", opacity=0.85),
                name="Incorrect",
                customdata=incorrect[["PitchCall", "Batter", "Pitcher", "Inning"]].fillna("").values if all(c in incorrect.columns for c in ["PitchCall","Batter","Pitcher","Inning"]) else None,
                hovertemplate="%{customdata[0]}<br>Batter: %{customdata[1]}<br>Pitcher: %{customdata[2]}<br>Inn: %{customdata[3]}<extra></extra>" if all(c in incorrect.columns for c in ["PitchCall","Batter","Pitcher","Inning"]) else None,
            ))
        add_strike_zone(fig)
        fig.update_layout(
            xaxis=dict(range=[-2.5, 2.5], showgrid=False, zeroline=False, title="", fixedrange=True, scaleanchor="y"),
            yaxis=dict(range=[0, 5], showgrid=False, zeroline=False, title="", fixedrange=True),
            height=420, plot_bgcolor="white", paper_bgcolor="white",
            font=dict(color="#1a1a2e", family="Inter, Arial, sans-serif"),
            margin=dict(l=20, r=10, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig, use_container_width=True, key="pg_ump_accuracy_scatter")

    with col_metrics:
        section_header("Accuracy Metrics")
        total = len(called)
        n_correct = called["Correct"].sum()
        n_strikes = is_strike.sum()
        n_balls = (~is_strike).sum()
        strike_correct = ((is_strike & iz).sum() / max(n_strikes, 1)) * 100
        ball_correct = ((~is_strike & ~iz).sum() / max(n_balls, 1)) * 100
        gifted = called["Gifted"].sum()
        missed = called["Missed"].sum()

        st.metric("Overall Accuracy", f"{n_correct / max(total, 1) * 100:.1f}%", f"{n_correct}/{total}")
        st.metric("Called Strike Accuracy", f"{strike_correct:.1f}%", f"{int((is_strike & iz).sum())}/{n_strikes}")
        st.metric("Called Ball Accuracy", f"{ball_correct:.1f}%", f"{int((~is_strike & ~iz).sum())}/{n_balls}")
        st.metric("Gifted Strikes", f"{int(gifted)}", help="Called strike outside zone")
        st.metric("Missed Strikes", f"{int(missed)}", help="Called ball inside zone")

    # ── 3d: Umpire's effective zone ──
    section_header("Umpire's Effective Zone")
    cs_locs = called[called["PitchCall"] == "StrikeCalled"]
    if len(cs_locs) >= 5:
        fig_ez = go.Figure()
        fig_ez.add_trace(go.Histogram2dContour(
            x=cs_locs["PlateLocSide"], y=cs_locs["PlateLocHeight"],
            colorscale=[[0, "rgba(255,255,255,0)"], [0.3, "rgba(200,60,60,0.3)"],
                        [0.6, "rgba(200,60,60,0.5)"], [1.0, "rgba(200,60,60,0.8)"]],
            showscale=False, ncontours=8,
            contours=dict(showlines=True, coloring="fill"),
            line=dict(width=0.5, color="rgba(150,150,150,0.3)"),
        ))
        fig_ez.add_shape(type="rect", x0=-ZONE_SIDE, x1=ZONE_SIDE,
                         y0=ZONE_HEIGHT_BOT, y1=ZONE_HEIGHT_TOP,
                         line=dict(color="#333", width=2, dash="dash"),
                         fillcolor="rgba(0,0,0,0)")
        fig_ez.update_layout(
            xaxis=dict(range=[-2.5, 2.5], showgrid=False, zeroline=False, title="", fixedrange=True, scaleanchor="y"),
            yaxis=dict(range=[0, 5], showgrid=False, zeroline=False, title="", fixedrange=True),
            height=380, plot_bgcolor="white", paper_bgcolor="white",
            font=dict(color="#1a1a2e", family="Inter, Arial, sans-serif"),
            margin=dict(l=20, r=10, t=30, b=20),
        )
        st.plotly_chart(fig_ez, use_container_width=True, key="pg_ump_eff_zone")
    else:
        st.info("Not enough called strikes for effective zone visualization.")

    # ── 3e: Breakdowns ──
    section_header("Breakdowns")
    bd_c1, bd_c2 = st.columns(2)

    # By count state
    with bd_c1:
        st.markdown("**By Count State** (Pitcher POV)")
        if "Balls" in called.columns and "Strikes" in called.columns:
            called["_CountState"] = called.apply(lambda r: _pg_count_state(r.get("Balls"), r.get("Strikes")), axis=1)
            rows = []
            for state in ["Ahead", "Even", "Behind"]:
                sub = called[called["_CountState"] == state]
                if sub.empty:
                    continue
                rows.append({
                    "Count": state, "Pitches": len(sub),
                    "Accuracy%": f"{sub['Correct'].mean()*100:.1f}",
                    "Gifted": int(sub["Gifted"].sum()),
                    "Missed": int(sub["Missed"].sum()),
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("Count data not available.")

    # By inning group
    with bd_c2:
        st.markdown("**By Inning Group**")
        if "Inning" in called.columns:
            called["_InnGrp"] = called["Inning"].apply(_pg_inning_group)
            rows = []
            for grp_name in ["Early (1-3)", "Mid (4-6)", "Late (7+)"]:
                sub = called[called["_InnGrp"] == grp_name]
                if sub.empty:
                    continue
                rows.append({
                    "Innings": grp_name, "Pitches": len(sub),
                    "Accuracy%": f"{sub['Correct'].mean()*100:.1f}",
                    "Gifted": int(sub["Gifted"].sum()),
                    "Missed": int(sub["Missed"].sum()),
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("Inning data not available.")

    # By batter side
    if "BatterSide" in called.columns:
        section_header("By Batter Side")
        bs_c1, bs_c2 = st.columns(2)
        for side, col in [("Right", bs_c1), ("Left", bs_c2)]:
            side_df = called[called["BatterSide"] == side]
            if side_df.empty:
                continue
            with col:
                st.markdown(f"**{side}-Handed Hitters** ({len(side_df)} calls, "
                            f"{side_df['Correct'].mean()*100:.1f}% accuracy)")
                fig_s = go.Figure()
                sc = side_df[side_df["Correct"]]
                si = side_df[~side_df["Correct"]]
                if not sc.empty:
                    fig_s.add_trace(go.Scatter(x=sc["PlateLocSide"], y=sc["PlateLocHeight"],
                                              mode="markers", marker=dict(size=6, color="#2ca02c", opacity=0.6),
                                              name="Correct", showlegend=False))
                if not si.empty:
                    fig_s.add_trace(go.Scatter(x=si["PlateLocSide"], y=si["PlateLocHeight"],
                                              mode="markers", marker=dict(size=8, color="#d62728", symbol="x", opacity=0.8),
                                              name="Incorrect", showlegend=False))
                add_strike_zone(fig_s)
                fig_s.update_layout(
                    xaxis=dict(range=[-2.5, 2.5], showgrid=False, zeroline=False, fixedrange=True, scaleanchor="y"),
                    yaxis=dict(range=[0, 5], showgrid=False, zeroline=False, fixedrange=True),
                    height=280, plot_bgcolor="white", paper_bgcolor="white",
                    margin=dict(l=10, r=10, t=10, b=10),
                    font=dict(color="#1a1a2e", family="Inter, Arial, sans-serif"),
                )
                st.plotly_chart(fig_s, use_container_width=True, key=f"pg_ump_side_{side}")

    # By pitcher
    if "Pitcher" in called.columns:
        section_header("By Pitcher")
        rows = []
        for p, sub in called.groupby("Pitcher"):
            rows.append({
                "Pitcher": display_name(p), "Pitches": len(sub),
                "Accuracy%": f"{sub['Correct'].mean()*100:.1f}",
                "Gifted": int(sub["Gifted"].sum()),
                "Missed": int(sub["Missed"].sum()),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values("Pitches", ascending=False), use_container_width=True, hide_index=True)

    # ── 3f: Shadow zone analysis ──
    section_header("Shadow Zone Analysis")
    shadow = (
        (called["PlateLocSide"].abs().between(ZONE_SIDE - BALL_RADIUS, ZONE_SIDE + BALL_RADIUS)) |
        (called["PlateLocHeight"].between(ZONE_HEIGHT_BOT - BALL_RADIUS, ZONE_HEIGHT_BOT + BALL_RADIUS)) |
        (called["PlateLocHeight"].between(ZONE_HEIGHT_TOP - BALL_RADIUS, ZONE_HEIGHT_TOP + BALL_RADIUS))
    )
    shadow_df = called[shadow]
    non_shadow_df = called[~shadow]
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric("Shadow Zone Pitches", len(shadow_df))
    with sc2:
        if len(shadow_df) > 0:
            st.metric("Shadow Zone Accuracy", f"{shadow_df['Correct'].mean()*100:.1f}%")
        else:
            st.metric("Shadow Zone Accuracy", "N/A")
    with sc3:
        if len(non_shadow_df) > 0:
            st.metric("Non-Shadow Accuracy", f"{non_shadow_df['Correct'].mean()*100:.1f}%")
        else:
            st.metric("Non-Shadow Accuracy", "N/A")

    # ── 3g: Impact metrics ──
    section_header("Impact Metrics — Missed Calls by Leverage")
    if "Balls" in called.columns and "Strikes" in called.columns:
        called["_Leverage"] = called.apply(lambda r: _pg_count_leverage(r.get("Balls"), r.get("Strikes")), axis=1)
        gifted_df = called[called["Gifted"]]
        missed_df = called[called["Missed"]]
        imp_rows = []
        for lev in ["High", "Medium", "Low"]:
            imp_rows.append({
                "Leverage": lev,
                "Gifted Strikes": int(gifted_df[gifted_df["_Leverage"] == lev].shape[0]),
                "Missed Strikes": int(missed_df[missed_df["_Leverage"] == lev].shape[0]),
            })
        st.dataframe(pd.DataFrame(imp_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("Count data not available for leverage analysis.")


# ── Pitcher Report ──
def _postgame_pitchers(gd, data):
    """Render the Pitcher Report tab for a single game."""
    section_header("Pitcher Report — Davidson Arms")

    dav_pitching = gd[gd["PitcherTeam"] == DAVIDSON_TEAM_ID].copy()
    if dav_pitching.empty:
        st.info("No Davidson pitching data for this game.")
        return

    pitchers = dav_pitching.groupby("Pitcher").size().sort_values(ascending=False).index.tolist()

    for idx, pitcher in enumerate(pitchers):
        pdf = dav_pitching[dav_pitching["Pitcher"] == pitcher].copy()
        n_pitches = len(pdf)
        jersey = JERSEY.get(pitcher, "")
        pos = POSITION.get(pitcher, "P")
        ip_est = _pg_estimate_ip(pdf)
        ks = len(pdf[pdf["KorBB"] == "Strikeout"]) if "KorBB" in pdf.columns else 0
        bbs = len(pdf[pdf["KorBB"] == "Walk"]) if "KorBB" in pdf.columns else 0
        hits = len(pdf[pdf["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"])]) if "PlayResult" in pdf.columns else 0

        player_header(pitcher, jersey, pos,
                      f"{n_pitches} pitches · ~{ip_est} IP",
                      f"K: {ks}  BB: {bbs}  H: {hits}")

        with st.expander(f"Details — {display_name(pitcher)}", expanded=(idx == 0)):
            _pg_pitcher_detail(pdf, data, pitcher)


def _pg_pitcher_detail(pdf, data, pitcher):
    """Render detailed pitcher breakdown inside an expander."""
    slug = _pg_slug(pitcher)
    col_left, col_right = st.columns(2)

    with col_left:
        # Pitch mix table
        section_header("Pitch Mix")
        if "TaggedPitchType" in pdf.columns:
            mix_rows = []
            total = len(pdf)
            for pt, grp in pdf.groupby("TaggedPitchType"):
                row = {"Pitch": pt, "N": len(grp), "Usage%": f"{len(grp)/total*100:.1f}"}
                if "RelSpeed" in grp.columns:
                    v = grp["RelSpeed"].dropna()
                    row["Avg Velo"] = f"{v.mean():.1f}" if len(v) > 0 else "-"
                    row["Max Velo"] = f"{v.max():.1f}" if len(v) > 0 else "-"
                if "SpinRate" in grp.columns:
                    s = grp["SpinRate"].dropna()
                    row["Avg Spin"] = f"{s.mean():.0f}" if len(s) > 0 else "-"
                if "InducedVertBreak" in grp.columns:
                    ivb = grp["InducedVertBreak"].dropna()
                    row["Avg IVB"] = f"{ivb.mean():.1f}" if len(ivb) > 0 else "-"
                if "HorzBreak" in grp.columns:
                    hb = grp["HorzBreak"].dropna()
                    row["Avg HB"] = f"{hb.mean():.1f}" if len(hb) > 0 else "-"
                mix_rows.append(row)
            if mix_rows:
                st.dataframe(pd.DataFrame(mix_rows).sort_values("N", ascending=False), use_container_width=True, hide_index=True)

        # Stuff+
        section_header("Stuff+")
        stuff = _compute_stuff_plus(pdf)
        if "StuffPlus" in stuff.columns and "TaggedPitchType" in stuff.columns:
            sp_summary = stuff.groupby("TaggedPitchType")["StuffPlus"].mean().round(0).reset_index()
            sp_summary.columns = ["Pitch", "Stuff+"]
            st.dataframe(sp_summary.sort_values("Stuff+", ascending=False), use_container_width=True, hide_index=True)

        # Command+
        section_header("Command+")
        cmd = _compute_command_plus(pdf, data)
        if not cmd.empty:
            st.dataframe(cmd, use_container_width=True, hide_index=True)

    with col_right:
        # Pitch location scatter
        section_header("Pitch Locations")
        loc = pdf.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        if not loc.empty and "TaggedPitchType" in loc.columns:
            fig_loc = go.Figure()
            for pt in sorted(loc["TaggedPitchType"].unique()):
                sub = loc[loc["TaggedPitchType"] == pt]
                color = PITCH_COLORS.get(pt, "#aaa")
                hover_data = []
                for _, row in sub.iterrows():
                    v = f"{row['RelSpeed']:.1f}" if pd.notna(row.get("RelSpeed")) else "?"
                    r = row.get("PitchCall", "?")
                    hover_data.append(f"{pt} {v}mph<br>{r}")
                fig_loc.add_trace(go.Scatter(
                    x=sub["PlateLocSide"], y=sub["PlateLocHeight"],
                    mode="markers", marker=dict(size=7, color=color, opacity=0.8,
                                                line=dict(width=0.5, color="white")),
                    name=pt, text=hover_data, hoverinfo="text",
                ))
            add_strike_zone(fig_loc)
            fig_loc.update_layout(
                xaxis=dict(range=[-2.5, 2.5], showgrid=False, zeroline=False, title="", fixedrange=True, scaleanchor="y"),
                yaxis=dict(range=[0, 5], showgrid=False, zeroline=False, title="", fixedrange=True),
                height=380, plot_bgcolor="white", paper_bgcolor="white",
                font=dict(color="#1a1a2e", family="Inter, Arial, sans-serif"),
                margin=dict(l=15, r=10, t=25, b=15),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
            )
            st.plotly_chart(fig_loc, use_container_width=True, key=f"pg_pit_loc_{slug}")

        # Movement profile
        section_header("Movement Profile")
        fig_mov = make_movement_profile(pdf, height=380)
        if fig_mov:
            st.plotly_chart(fig_mov, use_container_width=True, key=f"pg_pit_mov_{slug}")
        else:
            st.caption("Not enough movement data.")

    # Full width — Whiff & Chase table
    section_header("Whiff & Chase by Pitch Type")
    if "TaggedPitchType" in pdf.columns:
        wc_rows = []
        for pt, grp in pdf.groupby("TaggedPitchType"):
            n = len(grp)
            swings = grp[grp["PitchCall"].isin(SWING_CALLS)]
            whiffs = grp[grp["PitchCall"] == "StrikeSwinging"]
            loc_grp = grp.dropna(subset=["PlateLocSide", "PlateLocHeight"])
            if len(loc_grp) > 0:
                oz = ~in_zone_mask(loc_grp)
                oz_pitches = len(loc_grp[oz])
                oz_swings = loc_grp[oz & loc_grp["PitchCall"].isin(SWING_CALLS)]
                chase_pct = len(oz_swings) / max(oz_pitches, 1) * 100
            else:
                chase_pct = np.nan
            csw = grp["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"]).sum()
            wc_rows.append({
                "Pitch": pt, "Pitches": n,
                "Swings": len(swings),
                "Whiff%": f"{len(whiffs)/max(len(swings),1)*100:.1f}" if len(swings) > 0 else "-",
                "Chase%": f"{chase_pct:.1f}" if pd.notna(chase_pct) else "-",
                "CSW%": f"{csw/n*100:.1f}",
            })
        if wc_rows:
            st.dataframe(pd.DataFrame(wc_rows).sort_values("Pitches", ascending=False), use_container_width=True, hide_index=True)

    # Release point scatter
    if "RelSide" in pdf.columns and "RelHeight" in pdf.columns:
        section_header("Release Point")
        rel = pdf.dropna(subset=["RelSide", "RelHeight"])
        if not rel.empty and "TaggedPitchType" in rel.columns:
            fig_rel = go.Figure()
            for pt in sorted(rel["TaggedPitchType"].unique()):
                sub = rel[rel["TaggedPitchType"] == pt]
                color = PITCH_COLORS.get(pt, "#aaa")
                fig_rel.add_trace(go.Scatter(
                    x=sub["RelSide"], y=sub["RelHeight"],
                    mode="markers", marker=dict(size=6, color=color, opacity=0.7),
                    name=pt,
                ))
            fig_rel.update_layout(
                xaxis_title="Release Side (ft)", yaxis_title="Release Height (ft)",
                height=300, plot_bgcolor="white", paper_bgcolor="white",
                font=dict(color="#1a1a2e", family="Inter, Arial, sans-serif"),
                margin=dict(l=50, r=10, t=25, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
            )
            st.plotly_chart(fig_rel, use_container_width=True, key=f"pg_pit_rel_{slug}")

    # Key at-bats: K and BB
    section_header("Key At-Bats (K & BB)")
    if "KorBB" in pdf.columns:
        pa_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in pdf.columns]
        sort_cols = [c for c in ["Inning", "PAofInning", "PitchNo"] if c in pdf.columns]
        key_abs = pdf[pdf["KorBB"].isin(["Strikeout", "Walk"])]
        if not key_abs.empty and len(pa_cols) >= 2:
            for _, pa_key in key_abs.drop_duplicates(subset=pa_cols).iterrows():
                mask = pd.Series(True, index=pdf.index)
                for c in pa_cols:
                    mask = mask & (pdf[c] == pa_key[c])
                ab = pdf[mask].sort_values(sort_cols) if sort_cols else pdf[mask]
                if ab.empty:
                    continue
                result = pa_key.get("KorBB", "?")
                batter = display_name(pa_key["Batter"]) if "Batter" in pa_key.index else "?"
                inn = pa_key.get("Inning", "?")
                st.markdown(f"**Inn {inn}** vs {batter} — **{result}** ({len(ab)} pitches)")
                ab_c1, ab_c2 = st.columns([2, 1])
                with ab_c1:
                    st.caption(_pg_pitch_sequence_text(ab))
                with ab_c2:
                    fig_ab = _pg_mini_location_plot(ab)
                    if fig_ab:
                        ab_key = f"pg_pit_kab_{slug}_{inn}_{_pg_slug(str(batter))}"
                        st.plotly_chart(fig_ab, use_container_width=True, key=ab_key)


# ── Hitter Report ──
def _postgame_hitters(gd, data):
    """Render the Hitter Report tab for a single game."""
    section_header("Hitter Report — Davidson Bats")

    dav_hitting = gd[gd["BatterTeam"] == DAVIDSON_TEAM_ID].copy()
    if dav_hitting.empty:
        st.info("No Davidson batting data for this game.")
        return

    batters = dav_hitting.groupby("Batter").size().sort_values(ascending=False).index.tolist()

    for idx, batter in enumerate(batters):
        bdf = dav_hitting[dav_hitting["Batter"] == batter].copy()
        n_pitches = len(bdf)
        jersey = JERSEY.get(batter, "")
        pos = POSITION.get(batter, "")

        # Count PAs
        pa_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in bdf.columns]
        pa = bdf.drop_duplicates(subset=pa_cols).shape[0] if len(pa_cols) >= 2 else 0
        hits = len(bdf[bdf["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"])]) if "PlayResult" in bdf.columns else 0
        bbs = len(bdf[bdf["KorBB"] == "Walk"]) if "KorBB" in bdf.columns else 0
        ks = len(bdf[bdf["KorBB"] == "Strikeout"]) if "KorBB" in bdf.columns else 0
        bbe = len(bdf[(bdf["PitchCall"] == "InPlay") & bdf["ExitSpeed"].notna()]) if "ExitSpeed" in bdf.columns else 0

        player_header(batter, jersey, pos,
                      f"{n_pitches} pitches seen · {pa} PA",
                      f"H: {hits}  BB: {bbs}  K: {ks}  BBE: {bbe}")

        with st.expander(f"Details — {display_name(batter)}", expanded=(idx == 0)):
            _pg_hitter_detail(bdf, data, batter)


def _pg_hitter_detail(bdf, data, batter):
    """Render detailed hitter breakdown inside an expander."""
    slug = _pg_slug(batter)
    col_left, col_right = st.columns(2)

    with col_left:
        # Plate discipline metrics
        section_header("Plate Discipline")
        loc_df = bdf.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        if not loc_df.empty:
            iz = in_zone_mask(loc_df)
            in_zone_df = loc_df[iz]
            out_zone_df = loc_df[~iz]
            swings = loc_df[loc_df["PitchCall"].isin(SWING_CALLS)]
            iz_swings = in_zone_df[in_zone_df["PitchCall"].isin(SWING_CALLS)]
            oz_swings = out_zone_df[out_zone_df["PitchCall"].isin(SWING_CALLS)]
            whiffs = bdf[bdf["PitchCall"] == "StrikeSwinging"]
            iz_contacts = in_zone_df[in_zone_df["PitchCall"].isin(CONTACT_CALLS)]

            d_c1, d_c2 = st.columns(2)
            with d_c1:
                st.metric("Zone Swing%", f"{len(iz_swings)/max(len(in_zone_df),1)*100:.1f}%")
                st.metric("Chase%", f"{len(oz_swings)/max(len(out_zone_df),1)*100:.1f}%")
                st.metric("SwStr%", f"{len(whiffs)/max(len(bdf),1)*100:.1f}%")
            with d_c2:
                st.metric("Whiff%", f"{len(whiffs)/max(len(swings),1)*100:.1f}%" if len(swings) > 0 else "N/A")
                st.metric("Zone Contact%", f"{len(iz_contacts)/max(len(iz_swings),1)*100:.1f}%" if len(iz_swings) > 0 else "N/A")
        else:
            st.caption("No location data available.")

        # By pitch type table
        section_header("By Pitch Type")
        if "TaggedPitchType" in bdf.columns:
            pt_rows = []
            for pt, grp in bdf.groupby("TaggedPitchType"):
                n = len(grp)
                sw = grp[grp["PitchCall"].isin(SWING_CALLS)]
                wh = grp[grp["PitchCall"] == "StrikeSwinging"]
                loc_g = grp.dropna(subset=["PlateLocSide", "PlateLocHeight"])
                chase_pct = np.nan
                if len(loc_g) > 0:
                    oz_g = loc_g[~in_zone_mask(loc_g)]
                    if len(oz_g) > 0:
                        chase_pct = len(oz_g[oz_g["PitchCall"].isin(SWING_CALLS)]) / len(oz_g) * 100
                contact = grp[grp["PitchCall"].isin(CONTACT_CALLS)]
                avg_ev = contact["ExitSpeed"].mean() if "ExitSpeed" in contact.columns and len(contact) > 0 else np.nan
                pt_rows.append({
                    "Pitch": pt, "N": n,
                    "Swing%": f"{len(sw)/n*100:.1f}",
                    "Whiff%": f"{len(wh)/max(len(sw),1)*100:.1f}" if len(sw) > 0 else "-",
                    "Chase%": f"{chase_pct:.1f}" if pd.notna(chase_pct) else "-",
                    "Avg EV": f"{avg_ev:.1f}" if pd.notna(avg_ev) else "-",
                })
            if pt_rows:
                st.dataframe(pd.DataFrame(pt_rows).sort_values("N", ascending=False), use_container_width=True, hide_index=True)

    with col_right:
        # Batted ball quality
        section_header("Batted Ball Quality")
        in_play = bdf[(bdf["PitchCall"] == "InPlay")].copy()
        bbe_df = in_play.dropna(subset=["ExitSpeed"]) if "ExitSpeed" in in_play.columns else pd.DataFrame()
        if not bbe_df.empty:
            ev = bbe_df["ExitSpeed"]
            q_c1, q_c2 = st.columns(2)
            with q_c1:
                st.metric("Avg EV", f"{ev.mean():.1f} mph")
                st.metric("Max EV", f"{ev.max():.1f} mph")
                st.metric("BBE", len(bbe_df))
            with q_c2:
                if "Angle" in bbe_df.columns:
                    la = bbe_df["Angle"].dropna()
                    st.metric("Avg LA", f"{la.mean():.1f}°" if len(la) > 0 else "N/A")
                hh = (ev >= 95).mean() * 100
                st.metric("Hard Hit%", f"{hh:.1f}%")
                if "ExitSpeed" in bbe_df.columns and "Angle" in bbe_df.columns:
                    barrel = is_barrel_mask(bbe_df).mean() * 100
                    st.metric("Barrel%", f"{barrel:.1f}%")
        else:
            st.caption("No batted ball data.")

        # Spray chart
        section_header("Spray Chart")
        if not in_play.empty:
            fig_spray = make_spray_chart(in_play, height=320)
            if fig_spray:
                st.plotly_chart(fig_spray, use_container_width=True, key=f"pg_hit_spray_{slug}")
            else:
                st.caption("No spray chart data.")

    # Full width — Key at-bats
    section_header("Key At-Bats")
    pa_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in bdf.columns]
    sort_cols = [c for c in ["Inning", "PAofInning", "PitchNo"] if c in bdf.columns]
    if len(pa_cols) >= 2:
        # Identify key ABs: HR, XBH, K, BB, long PAs (6+ pitches)
        pa_groups = []
        for pa_key, ab in bdf.groupby(pa_cols[1:]):  # group within game
            if not isinstance(pa_key, tuple):
                pa_key = (pa_key,)
            ab_sorted = ab.sort_values(sort_cols) if sort_cols else ab
            is_key = False
            result_label = ""
            if "PlayResult" in ab.columns:
                if ab["PlayResult"].eq("HomeRun").any():
                    is_key, result_label = True, "HR"
                elif ab["PlayResult"].isin(["Double", "Triple"]).any():
                    is_key, result_label = True, ab[ab["PlayResult"].isin(["Double", "Triple"])]["PlayResult"].iloc[0]
            if "KorBB" in ab.columns:
                if ab["KorBB"].eq("Strikeout").any():
                    is_key, result_label = True, result_label or "K"
                elif ab["KorBB"].eq("Walk").any():
                    is_key, result_label = True, result_label or "BB"
            if len(ab) >= 6:
                is_key = True
                result_label = result_label or f"{len(ab)}-pitch PA"
            if is_key:
                pa_groups.append((ab_sorted, result_label))

        if pa_groups:
            for ab, result_label in pa_groups:
                inn = ab.iloc[0].get("Inning", "?")
                pitcher_name = display_name(ab.iloc[0]["Pitcher"]) if "Pitcher" in ab.columns else "?"
                st.markdown(f"**Inn {inn}** vs {pitcher_name} — **{result_label}** ({len(ab)} pitches)")
                ab_c1, ab_c2 = st.columns([2, 1])
                with ab_c1:
                    st.caption(_pg_pitch_sequence_text(ab))
                with ab_c2:
                    fig_ab = _pg_mini_location_plot(ab)
                    if fig_ab:
                        ab_key = f"pg_hit_kab_{slug}_{inn}_{_pg_slug(str(ab.iloc[0].get('Pitcher','')))}"
                        st.plotly_chart(fig_ab, use_container_width=True, key=ab_key)
        else:
            st.caption("No notable at-bats (HR, XBH, K, BB, or 6+ pitch PA).")
    else:
        st.caption("PA identification columns not available.")


# ── Main postgame page ──
def page_postgame(data):
    """Postgame Report page — single game selector with Umpire, Pitcher, Hitter tabs."""
    st.title("Postgame Report")

    # Filter to Davidson games
    dav = data[(data["PitcherTeam"] == DAVIDSON_TEAM_ID) | (data["BatterTeam"] == DAVIDSON_TEAM_ID)]
    if dav.empty:
        st.warning("No Davidson game data available.")
        return

    # Build game list
    games = dav.groupby(["Date", "GameID"]).agg(
        Home=("HomeTeam", "first"),
        Away=("AwayTeam", "first"),
        Pitches=("PitchNo", "count"),
    ).reset_index().sort_values("Date", ascending=False)

    if games.empty:
        st.warning("No games found.")
        return

    game_labels = {}
    for _, row in games.iterrows():
        dt = row["Date"].strftime("%Y-%m-%d") if pd.notna(row["Date"]) else "?"
        game_labels[row["GameID"]] = f"{dt}  {row['Away']} @ {row['Home']}  ({row['Pitches']} pitches)"

    sel_game = st.selectbox("Select Game", games["GameID"].tolist(),
                            format_func=lambda g: game_labels.get(g, str(g)),
                            key="pg_game_select")

    gd = data[data["GameID"] == sel_game]
    if gd.empty:
        st.warning("No data for selected game.")
        return

    tab_ump, tab_pit, tab_hit = st.tabs(["Umpire Report", "Pitcher Report", "Hitter Report"])
    with tab_ump:
        _postgame_umpire(gd)
    with tab_pit:
        _postgame_pitchers(gd, data)
    with tab_hit:
        _postgame_hitters(gd, data)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    _logo_sidebar = os.path.join(_APP_DIR, "logo_real.png")
    if os.path.exists(_logo_sidebar):
        _lcol1, _lcol2, _lcol3 = st.sidebar.columns([1, 2, 1])
        with _lcol2:
            st.image(_logo_sidebar, use_container_width=True)
    st.sidebar.markdown(
        '<div style="text-align:center;padding:2px 0 5px 0;">'
        '<span style="display:block;font-size:20px;font-weight:800;font-family:Inter,sans-serif;letter-spacing:1px;color:#cc0000 !important;">W.I.L.D.C.A.T.S.</span>'
        '<span style="display:block;font-size:10px;letter-spacing:1px;text-transform:uppercase;color:#ccc !important;'
        'font-family:Inter,sans-serif;">Davidson Baseball Analytics</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    page = st.sidebar.radio("Navigation", [
        "Team Overview",
        "Hitting",
        "Pitching",
        "Catcher Analytics",
        "Player Development",
        "Defensive Positioning",
        "Opponent Scouting",
        "Postgame Report",
    ], label_visibility="collapsed")

    data = load_davidson_data()
    if data.empty:
        st.error("No data loaded.")
        return

    st.sidebar.markdown("---")
    _sb = get_sidebar_stats()
    st.sidebar.markdown(f'<div style="font-size:12px;color:#888 !important;padding:0 10px;">'
                        f'<b style="color:#cc0000 !important;">{_sb["total_pitches"]:,}</b> pitches<br>'
                        f'<b style="color:#cc0000 !important;">{_sb["n_seasons"]}</b> seasons '
                        f'({_sb["min_season"]}-{_sb["max_season"]})<br>'
                        f'<b style="color:#cc0000 !important;">{_sb["n_dav_games"]}</b> Davidson games<br>'
                        f'<b style="color:#cc0000 !important;">{len(ROSTER_2026)}</b> rostered players<br>'
                        f'<b style="color:#cc0000 !important;">{_sb["n_pitchers"]:,}</b> pitchers in DB<br>'
                        f'<b style="color:#cc0000 !important;">{_sb["n_batters"]:,}</b> hitters in DB'
                        f'</div>', unsafe_allow_html=True)

    if page == "Hitting":
        page_hitting(data)
    elif page == "Pitching":
        page_pitching(data)
    elif page == "Catcher Analytics":
        page_catcher(data)
    elif page == "Team Overview":
        page_team(data)
    elif page == "Player Development":
        page_development(data)
    elif page == "Defensive Positioning":
        page_defensive_positioning(data)
    elif page == "Opponent Scouting":
        page_scouting(data)
    elif page == "Postgame Report":
        page_postgame(data)


if __name__ == "__main__":
    main()
