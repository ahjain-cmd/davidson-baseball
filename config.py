"""
Davidson Baseball Analytics — Configuration & Constants.

All roster data, pitch mappings, color definitions, zone constants,
and name normalization utilities live here.
"""
import os
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("TRACKMAN_CSV_ROOT", os.path.join(_APP_DIR, "v3"))
PARQUET_FIXED_PATH = os.path.join(_APP_DIR, "all_trackman_fixed.parquet")
PARQUET_PATH = PARQUET_FIXED_PATH if os.path.exists(PARQUET_FIXED_PATH) else os.path.join(_APP_DIR, "all_trackman.parquet")
DUCKDB_PATH = os.path.join(_APP_DIR, "davidson.duckdb")
DAVIDSON_TEAM_ID = "DAV_WIL"

# ── TrueMedia API credentials ────────────────
TM_USERNAME = "frhowden@davidson.edu"
TM_SITENAME = "davidson-ncaabaseball"
TM_MASTER_TOKEN = os.environ.get(
    "TM_MASTER_TOKEN",
    "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyIjoiZjZlZWEwYzViZmUwZTY4ZmEwZDUyMGQyMDU2NTNmYzciLCJpYXQiOjE3NzAwMDM4NTd9.c2QwNDh0Sy54ystStrYvORy4PrEQEJbUFDAacCH55EA",
)
CACHE_DIR = os.path.join(_APP_DIR, ".cache")
TUNNEL_BENCH_PATH = os.path.join(CACHE_DIR, "tunnel_benchmarks.json")
TUNNEL_WEIGHTS_PATH = os.path.join(CACHE_DIR, "tunnel_weights.json")

# ── Roster ─────────────────────────────────────
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

# ── Bryant 2026 Roster ────────────────────────
BRYANT_TEAM_NAME = "Bryant University"
BRYANT_COMBINED_TEAM_ID = "BRYANT_COMBINED_2026"

# Transfers: "Last, First" -> previous school search name.
# Players NOT in this dict are assumed to have been at Bryant for 2024-2025.
BRYANT_TRANSFERS = {
    "Wensley, Casey": "Wheaton College (Massachusetts)",
    "Greger, Gavin": "University of Connecticut",
    "Story, Hudson": "LA Mission",  # community college — not in TrueMedia
    "Galusha, Thomas": "University of Connecticut",
    "Irizarry, Carlos": "Penn State Harrisburg",
    "Scudder, Dylan": "Eastern Connecticut",
    "Garcia, Ellis": "West Virginia",
    "Dressler, Justin": "Pace",
    "Vining, Aidan": "Johnson and Wales",
    "Durand, Brandyn": "Chipola",
    "Salsberg, Zev": "Ohio State",
    "Flaherty, Tommy": "Clark University",
    "Schiff, Cole": "UNC Asheville",
    "White, Landon": "Ithaca",
}
BRYANT_ROSTER_2026 = {
    "Vazquez, Alejandro", "Kingsbury, Hunter", "Prince, Dylan", "Zyons, Zac",
    "Belcher, Michael", "Wensley, Casey", "Greger, Gavin", "Ferrell, Vince",
    "Papetti, Cam", "Fiatarone, Mike", "Carter, Ian", "Hilburger, Kaden",
    "Story, Hudson", "Hackett, Justin", "Gaudreau, Jacob", "Davis, Ty",
    "Gorman, Greg", "Galusha, Thomas", "Saul, Charlie", "Irizarry, Carlos",
    "Scudder, Dylan", "Garcia, Ellis", "Dressler, Justin", "Hurley, Will",
    "Burkholz, Max", "Vining, Aidan", "Durand, Brandyn", "Soroko, Cameron",
    "Birchard, Owen", "Malloy, Tommy", "Davis, Zach", "Vanesko, Jackson",
    "Perez, Yamil", "Zaslaw, Sean", "Lewis, Bradley", "Salsberg, Zev",
    "Flaherty, Tommy", "Schiff, Cole", "White, Landon", "Mulholland, Billy",
    "Maher, Thomas", "Dobis, Jameson", "Clifford, Sean",
}

BRYANT_JERSEY = {
    "Vazquez, Alejandro": 1, "Kingsbury, Hunter": 2, "Prince, Dylan": 3,
    "Zyons, Zac": 4, "Belcher, Michael": 5, "Wensley, Casey": 6,
    "Greger, Gavin": 7, "Ferrell, Vince": 9, "Papetti, Cam": 10,
    "Fiatarone, Mike": 11, "Carter, Ian": 12, "Hilburger, Kaden": 13,
    "Story, Hudson": 14, "Hackett, Justin": 15, "Gaudreau, Jacob": 16,
    "Davis, Ty": 17, "Gorman, Greg": 18, "Galusha, Thomas": 19,
    "Saul, Charlie": 20, "Irizarry, Carlos": 21, "Scudder, Dylan": 22,
    "Garcia, Ellis": 23, "Dressler, Justin": 24, "Hurley, Will": 25,
    "Burkholz, Max": 26, "Vining, Aidan": 27, "Durand, Brandyn": 28,
    "Soroko, Cameron": 29, "Birchard, Owen": 30, "Malloy, Tommy": 31,
    "Davis, Zach": 32, "Vanesko, Jackson": 33, "Perez, Yamil": 34,
    "Zaslaw, Sean": 35, "Lewis, Bradley": 36, "Salsberg, Zev": 37,
    "Flaherty, Tommy": 39, "Schiff, Cole": 40, "White, Landon": 41,
    "Mulholland, Billy": 42, "Maher, Thomas": 43, "Dobis, Jameson": 50,
}

BRYANT_POSITION = {
    "Vazquez, Alejandro": "NF", "Kingsbury, Hunter": "OF", "Prince, Dylan": "INF",
    "Zyons, Zac": "INF", "Belcher, Michael": "RHP", "Wensley, Casey": "INF",
    "Greger, Gavin": "OF", "Ferrell, Vince": "OF", "Papetti, Cam": "C/INF",
    "Fiatarone, Mike": "INF", "Carter, Ian": "RHP", "Hilburger, Kaden": "OF/RHP",
    "Story, Hudson": "INF", "Hackett, Justin": "INF", "Gaudreau, Jacob": "C",
    "Davis, Ty": "RHP", "Gorman, Greg": "OF", "Galusha, Thomas": "RHP",
    "Saul, Charlie": "OF", "Irizarry, Carlos": "INF", "Scudder, Dylan": "RHP",
    "Garcia, Ellis": "INF", "Dressler, Justin": "RHP", "Hurley, Will": "RHP",
    "Burkholz, Max": "OF", "Vining, Aidan": "LHP", "Durand, Brandyn": "C",
    "Soroko, Cameron": "INF/OF", "Birchard, Owen": "RHP", "Malloy, Tommy": "C",
    "Davis, Zach": "RHP", "Vanesko, Jackson": "LHP", "Perez, Yamil": "C",
    "Zaslaw, Sean": "RHP", "Lewis, Bradley": "LHP", "Salsberg, Zev": "RHP",
    "Flaherty, Tommy": "RHP", "Schiff, Cole": "LHP", "White, Landon": "RHP",
    "Mulholland, Billy": "RHP/C", "Maher, Thomas": "LHP", "Dobis, Jameson": "RHP",
    "Clifford, Sean": "RHP",
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

# ── Pitch colors & mappings ───────────────────
PITCH_COLORS = {
    "Fastball": "#d22d49", "Sinker": "#fe6100", "Cutter": "#933f8e",
    "Slider": "#f7c631", "Curveball": "#00d1ed", "Changeup": "#1dbe3a",
    "Splitter": "#c99b6e", "Knuckle Curve": "#2d7fc1", "Sweeper": "#dbab00",
    "Other": "#aaaaaa",
}

PITCH_TYPE_MAP = {
    "FourSeamFastBall": "Fastball",
    "OneSeamFastBall": "Sinker",
    "TwoSeamFastBall": "Sinker",
    "ChangeUp": "Changeup",
    "Knuckleball": "Other",
    "Undefined": "Other",
    "UN": "Other",
    "Unknown": "Other",
    "UNK": "Other",
}
PITCH_TYPES_TO_DROP = {"Other", "Undefined", "UN", "Unknown", "UNK"}

TM_PITCH_PCT_COLS = {
    "4Seam%": "Fastball", "Sink2Seam%": "Sinker", "Cutter%": "Cutter",
    "Slider%": "Slider", "Curve%": "Curveball", "Change%": "Changeup",
    "Split%": "Splitter", "Sweeper%": "Sweeper",
}

# ── Strike zone constants ──────────────────────
ZONE_SIDE = 0.83
ZONE_HEIGHT_BOT = 1.5
ZONE_HEIGHT_TOP = 3.5
MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE = 20
PLATE_SIDE_MAX = 2.5
PLATE_HEIGHT_MIN = 0.0
PLATE_HEIGHT_MAX = 5.5
MIN_PITCH_USAGE_PCT = 5.0

# ── Swing / contact call lists ─────────────────
SWING_CALLS = ["StrikeSwinging", "FoulBall", "FoulBallNotFieldable", "FoulBallFieldable", "InPlay"]
CONTACT_CALLS = ["FoulBall", "FoulBallNotFieldable", "FoulBallFieldable", "InPlay"]

# SQL-ready versions
_SWING_CALLS_SQL = "('StrikeSwinging','FoulBall','FoulBallNotFieldable','FoulBallFieldable','InPlay')"
_CONTACT_CALLS_SQL = "('FoulBall','FoulBallNotFieldable','FoulBallFieldable','InPlay')"
_IZ_COND = "ABS(PlateLocSide) <= 0.83 AND PlateLocHeight BETWEEN 1.5 AND 3.5"
_HAS_LOC = "PlateLocSide IS NOT NULL AND PlateLocHeight IS NOT NULL"
_OZ_COND = f"NOT ({_IZ_COND}) AND {_HAS_LOC}"

# ── Stuff+ weights per pitch type ──────────────
STUFF_WEIGHTS = {
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
STUFF_WEIGHTS_DEFAULT = {"RelSpeed": 1.0, "InducedVertBreak": 1.0, "HorzBreak": 1.0, "Extension": 1.0, "VertApprAngle": 1.0, "SpinRate": 1.0}


# ── Utility functions ──────────────────────────

def _norm_name_sql(col):
    """Normalize whitespace and comma spacing in SQL."""
    return (
        f"regexp_replace("
        f"regexp_replace("
        f"regexp_replace(trim({col}), '\\\\s+', ' '),"
        f"'\\\\s+,', ','),"
        f"',\\\\s*', ', ')"
    )


def _name_case_sql(col):
    """SQL CASE expression that normalizes player names to match NAME_MAP."""
    norm = _norm_name_sql(col)
    def _esc(s):
        return s.replace("'", "''")
    parts = " ".join(f"WHEN {norm} = '{_esc(old)}' THEN '{_esc(new)}'" for old, new in NAME_MAP.items())
    return f"CASE {parts} ELSE {norm} END"


def _name_sql(col):
    """Alias for _name_case_sql."""
    return _name_case_sql(col)


def _normalize_hand(series):
    """Normalize handedness to Left/Right; others become NA."""
    s = series.astype(str).str.strip()
    s = s.replace({"L": "Left", "R": "Right", "B": "Both"})
    s = s.where(s.isin(["Left", "Right"]))
    return s


def safe_mode(series, default=""):
    """Return the mode of a Series, or *default* if no mode exists."""
    m = series.mode()
    return m.iloc[0] if len(m) > 0 else default


def is_barrel(ev, la):
    """Statcast barrel definition."""
    if pd.isna(ev) or pd.isna(la):
        return False
    if ev < 98:
        return False
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


def in_zone_mask(df, batter_zones=None, batter_col="Batter"):
    """Per-pitch boolean mask: True if pitch is inside the batter's strike zone."""
    valid_loc = (
        df["PlateLocSide"].between(-PLATE_SIDE_MAX, PLATE_SIDE_MAX) &
        df["PlateLocHeight"].between(PLATE_HEIGHT_MIN, PLATE_HEIGHT_MAX)
    )
    side_ok = df["PlateLocSide"].abs() <= ZONE_SIDE
    if batter_zones and batter_col in df.columns:
        bot = df[batter_col].map(lambda b: batter_zones.get(b, (ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP))[0])
        top = df[batter_col].map(lambda b: batter_zones.get(b, (ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP))[1])
        height_ok = (df["PlateLocHeight"] >= bot) & (df["PlateLocHeight"] <= top)
    else:
        height_ok = df["PlateLocHeight"].between(ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP)
    return (valid_loc & side_ok & height_ok).fillna(False)


def normalize_pitch_types(df):
    """Normalize pitch type names and null out junk/undefined."""
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
    df = df[df["TaggedPitchType"].notna()]
    total = len(df)
    if total == 0:
        return df
    counts = df["TaggedPitchType"].value_counts()
    keep = counts[counts / total * 100 >= min_pct].index
    return df[df["TaggedPitchType"].isin(keep)]


def _is_position_player(name):
    """True if player is a position player (not a pure pitcher)."""
    pos = POSITION.get(name, "")
    return pos not in ("RHP", "LHP")


def filter_davidson(data, role="pitcher"):
    if role == "pitcher":
        return data[(data["PitcherTeam"] == DAVIDSON_TEAM_ID) & (data["Pitcher"].isin(ROSTER_2026))].copy()
    else:
        return data[(data["BatterTeam"] == DAVIDSON_TEAM_ID) & (data["Batter"].isin(ROSTER_2026))].copy()


def display_name(name, escape_html=True):
    import html as html_mod
    if not name:
        return "Unknown"
    parts = name.split(", ")
    result = f"{parts[1]} {parts[0]}" if len(parts) == 2 else name
    return html_mod.escape(result) if escape_html else result


def tm_name_to_trackman(full_name):
    """Convert TrueMedia 'First Last' or 'First Last Jr.' to Trackman 'Last, First' / 'Last, First Jr.'."""
    if not full_name or not isinstance(full_name, str):
        return full_name
    parts = full_name.strip().split()
    if len(parts) < 2:
        return full_name
    suffixes = {"Jr.", "Jr", "Sr.", "Sr", "II", "III", "IV", "V"}
    suffix = ""
    if parts[-1] in suffixes:
        suffix = " " + parts[-1]
        parts = parts[:-1]
    if len(parts) < 2:
        return full_name
    first = parts[0]
    last = " ".join(parts[1:])
    return f"{last}, {first}{suffix}"


def get_percentile(value, series):
    from scipy.stats import percentileofscore
    if pd.isna(value) or series.dropna().empty:
        return np.nan
    return percentileofscore(series.dropna(), value, kind='rank')
