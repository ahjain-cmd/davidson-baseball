import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import glob
import os
import numpy as np
from scipy.stats import percentileofscore

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("TRACKMAN_CSV_ROOT", os.path.join(_APP_DIR, "..", "v3"))
PARQUET_PATH = os.path.join(_APP_DIR, "all_trackman.parquet")
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
    "Daly, Jameson": "Daly, Jamie",
    "Hall, Edward": "Hall, Ed",
    "Hamilton, Matthew": "Hamilton, Matt",
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
    "OneSeamFastBall": "Fastball",
    "TwoSeamFastBall": "Sinker",
    "ChangeUp": "Changeup",
    "Knuckleball": "Other",
    "Undefined": "Other",
}
PITCH_TYPES_TO_DROP = {"Other", "Undefined"}


def normalize_pitch_types(df):
    """Normalize pitch type names and drop junk/undefined."""
    if "TaggedPitchType" not in df.columns:
        return df
    df = df.copy()
    df["TaggedPitchType"] = df["TaggedPitchType"].replace(PITCH_TYPE_MAP)
    df = df[~df["TaggedPitchType"].isin(PITCH_TYPES_TO_DROP)]
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
@st.cache_data(show_spinner="Loading Trackman data...")
def load_all_data():
    if os.path.exists(PARQUET_PATH):
        data = pd.read_parquet(PARQUET_PATH)
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        if "Season" in data.columns:
            data["Season"] = pd.to_numeric(data["Season"], errors="coerce").astype("Int64")
        data = normalize_pitch_types(data)
        return data
    all_csvs = sorted(glob.glob(os.path.join(DATA_ROOT, "**/CSV/*.csv"), recursive=True))
    all_csvs = [f for f in all_csvs if "positioning" not in f]
    frames = []
    for fp in all_csvs:
        try:
            frames.append(pd.read_csv(fp, low_memory=False))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Season"] = data["Date"].dt.year
    for col in ["Pitcher", "Batter"]:
        data[col] = data[col].astype(str).str.strip().replace(NAME_MAP)
    for c in ["RelSpeed", "SpinRate", "InducedVertBreak", "HorzBreak",
              "PlateLocHeight", "PlateLocSide", "ExitSpeed", "Angle",
              "Direction", "Distance", "Extension", "RelHeight", "RelSide",
              "VertApprAngle", "HorzApprAngle", "SpinAxis", "VertBreak",
              "ZoneSpeed", "EffectiveVelo", "HangTime", "PopTime"]:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")
    data = normalize_pitch_types(data)
    return data


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


@st.cache_data(show_spinner=False)
def compute_batter_stats(data, season_filter=None):
    df = data.copy()
    if season_filter:
        df = df[df["Season"].isin(season_filter)]
    in_zone = (df["PlateLocSide"].abs() <= 0.83) & (df["PlateLocHeight"].between(1.5, 3.5))
    out_zone = ~in_zone & df["PlateLocSide"].notna() & df["PlateLocHeight"].notna()
    is_swing = df["PitchCall"].isin(SWING_CALLS)
    is_whiff = df["PitchCall"] == "StrikeSwinging"
    is_contact = df["PitchCall"].isin(CONTACT_CALLS)
    rows = []
    for (batter, team), grp in df.groupby(["Batter", "BatterTeam"]):
        pa = int(grp["PitchofPA"].eq(1).sum())
        if pa < 5:
            continue
        ip = grp[grp["PitchCall"] == "InPlay"]
        batted = ip.dropna(subset=["ExitSpeed"])
        swings = grp[is_swing.reindex(grp.index, fill_value=False)]
        whiffs = grp[is_whiff.reindex(grp.index, fill_value=False)]
        contacts = grp[is_contact.reindex(grp.index, fill_value=False)]
        grp_in_zone = grp[in_zone.reindex(grp.index, fill_value=False)]
        grp_out_zone = grp[out_zone.reindex(grp.index, fill_value=False)]
        swings_out = grp_out_zone[grp_out_zone["PitchCall"].isin(SWING_CALLS)]
        swings_in = grp_in_zone[grp_in_zone["PitchCall"].isin(SWING_CALLS)]
        contacts_in = grp_in_zone[grp_in_zone["PitchCall"].isin(CONTACT_CALLS)]
        contacts_out = grp_out_zone[grp_out_zone["PitchCall"].isin(CONTACT_CALLS)]
        n = len(batted)
        hard = len(batted[batted["ExitSpeed"] >= 95]) if n > 0 else 0
        barrels = len(batted[(batted["ExitSpeed"] >= 98) & (batted["Angle"].between(8, 32))]) if n > 0 else 0
        la_sweet = len(batted[batted["Angle"].between(8, 32)]) if n > 0 else 0
        ks = len(grp[grp["KorBB"] == "Strikeout"])
        bbs = len(grp[grp["KorBB"] == "Walk"])

        # Batted ball profile
        gb = len(batted[batted["TaggedHitType"] == "GroundBall"]) if n > 0 else 0
        fb = len(batted[batted["TaggedHitType"] == "FlyBall"]) if n > 0 else 0
        ld = len(batted[batted["TaggedHitType"] == "LineDrive"]) if n > 0 else 0
        pu = len(batted[batted["TaggedHitType"] == "Popup"]) if n > 0 else 0

        # Directional
        pull_mask = batted["Direction"].notna()
        if pull_mask.any():
            side = grp["BatterSide"].mode().iloc[0] if grp["BatterSide"].notna().any() else "Right"
            if side == "Left":
                pull = len(batted[batted["Direction"] > 15])
                oppo = len(batted[batted["Direction"] < -15])
            else:
                pull = len(batted[batted["Direction"] < -15])
                oppo = len(batted[batted["Direction"] > 15])
            straight = n - pull - oppo
        else:
            pull = oppo = straight = 0

        rows.append({
            "Batter": batter, "BatterTeam": team, "PA": pa, "BBE": n,
            "AvgEV": batted["ExitSpeed"].mean() if n > 0 else np.nan,
            "MaxEV": batted["ExitSpeed"].max() if n > 0 else np.nan,
            "HardHitPct": hard / n * 100 if n > 0 else np.nan,
            "Barrels": barrels,
            "BarrelPct": barrels / n * 100 if n > 0 else np.nan,
            "BarrelPA": barrels / pa * 100 if pa > 0 else np.nan,
            "SweetSpotPct": la_sweet / n * 100 if n > 0 else np.nan,
            "AvgLA": batted["Angle"].mean() if n > 0 else np.nan,
            "AvgDist": batted["Distance"].mean() if n > 0 and batted["Distance"].notna().any() else np.nan,
            "WhiffPct": len(whiffs) / max(len(swings), 1) * 100,
            "KPct": ks / pa * 100,
            "BBPct": bbs / pa * 100,
            "ChasePct": len(swings_out) / max(len(grp_out_zone), 1) * 100,
            "ChaseContact": len(contacts_out) / max(len(swings_out), 1) * 100 if len(swings_out) > 0 else np.nan,
            "ZoneSwingPct": len(swings_in) / max(len(grp_in_zone), 1) * 100,
            "ZoneContactPct": len(contacts_in) / max(len(swings_in), 1) * 100 if len(swings_in) > 0 else np.nan,
            "ZonePct": len(grp_in_zone) / max(len(grp[grp["PlateLocSide"].notna()]), 1) * 100,
            "SwingPct": len(swings) / max(len(grp), 1) * 100,
            "GBPct": gb / n * 100 if n > 0 else np.nan,
            "FBPct": fb / n * 100 if n > 0 else np.nan,
            "LDPct": ld / n * 100 if n > 0 else np.nan,
            "PUPct": pu / n * 100 if n > 0 else np.nan,
            "AirPct": (fb + ld + pu) / n * 100 if n > 0 else np.nan,
            "PullPct": pull / n * 100 if n > 0 else np.nan,
            "StraightPct": straight / n * 100 if n > 0 else np.nan,
            "OppoPct": oppo / n * 100 if n > 0 else np.nan,
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def compute_pitcher_stats(data, season_filter=None):
    df = data.copy()
    if season_filter:
        df = df[df["Season"].isin(season_filter)]
    in_zone = (df["PlateLocSide"].abs() <= 0.83) & (df["PlateLocHeight"].between(1.5, 3.5))
    out_zone = ~in_zone & df["PlateLocSide"].notna() & df["PlateLocHeight"].notna()
    is_swing = df["PitchCall"].isin(SWING_CALLS)
    is_contact = df["PitchCall"].isin(CONTACT_CALLS)
    rows = []
    for (pitcher, team), grp in df.groupby(["Pitcher", "PitcherTeam"]):
        if len(grp) < 20:
            continue
        pa = int(grp["PitchofPA"].eq(1).sum())
        fb = grp[grp["TaggedPitchType"].isin(["Fastball", "Sinker", "Cutter"])]
        swings = grp[is_swing.reindex(grp.index, fill_value=False)]
        whiffs = grp[grp["PitchCall"] == "StrikeSwinging"]
        grp_in_zone = grp[in_zone.reindex(grp.index, fill_value=False)]
        grp_out_zone = grp[out_zone.reindex(grp.index, fill_value=False)]
        swings_out = grp_out_zone[grp_out_zone["PitchCall"].isin(SWING_CALLS)]
        contacts_in = grp_in_zone[grp_in_zone["PitchCall"].isin(CONTACT_CALLS)]
        swings_in = grp_in_zone[grp_in_zone["PitchCall"].isin(SWING_CALLS)]
        ip = grp[grp["PitchCall"] == "InPlay"]
        batted = ip.dropna(subset=["ExitSpeed"])
        ks = len(grp[grp["KorBB"] == "Strikeout"])
        bbs = len(grp[grp["KorBB"] == "Walk"])
        n_batted = len(batted)
        gb = len(ip[ip["TaggedHitType"] == "GroundBall"]) if len(ip) > 0 else 0
        hard = len(batted[batted["ExitSpeed"] >= 95]) if n_batted > 0 else 0
        barrels = len(batted[(batted["ExitSpeed"] >= 98) & (batted["Angle"].between(8, 32))]) if n_batted > 0 else 0
        rows.append({
            "Pitcher": pitcher, "PitcherTeam": team, "Pitches": len(grp), "PA": pa,
            "AvgFBVelo": fb["RelSpeed"].mean() if len(fb) > 0 and fb["RelSpeed"].notna().any() else np.nan,
            "MaxFBVelo": fb["RelSpeed"].max() if len(fb) > 0 and fb["RelSpeed"].notna().any() else np.nan,
            "AvgSpin": grp["SpinRate"].mean() if grp["SpinRate"].notna().any() else np.nan,
            "Extension": grp["Extension"].mean() if grp["Extension"].notna().any() else np.nan,
            "WhiffPct": len(whiffs) / max(len(swings), 1) * 100,
            "KPct": ks / max(pa, 1) * 100,
            "BBPct": bbs / max(pa, 1) * 100,
            "ZonePct": len(grp_in_zone) / max(len(grp[grp["PlateLocSide"].notna()]), 1) * 100,
            "ChasePct": len(swings_out) / max(len(grp_out_zone), 1) * 100,
            "ZoneContactPct": len(contacts_in) / max(len(swings_in), 1) * 100 if len(swings_in) > 0 else np.nan,
            "AvgEVAgainst": batted["ExitSpeed"].mean() if n_batted > 0 else np.nan,
            "HardHitAgainst": hard / max(n_batted, 1) * 100 if n_batted > 0 else np.nan,
            "BarrelPctAgainst": barrels / max(n_batted, 1) * 100 if n_batted > 0 else np.nan,
            "GBPct": gb / max(len(ip), 1) * 100 if len(ip) > 0 else np.nan,
            "SwingPct": len(swings) / max(len(grp), 1) * 100,
        })
    return pd.DataFrame(rows)


def get_percentile(value, series):
    if pd.isna(value) or series.dropna().empty:
        return np.nan
    return percentileofscore(series.dropna(), value, kind='rank')


def display_name(name):
    parts = name.split(", ")
    return f"{parts[1]} {parts[0]}" if len(parts) == 2 else name


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


def render_savant_percentile_section(metrics_data, title=None):
    """Render Baseball Savant style percentile ranking section.
    metrics_data: list of (label, value, percentile, fmt, higher_is_better)
    """
    if title:
        st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

    # POOR / AVERAGE / GREAT legend
    st.markdown(
        '<div style="display:flex;justify-content:space-between;margin-bottom:8px;padding:0 4px;">'
        '<span style="font-size:10px;font-weight:700;color:#14365d !important;letter-spacing:0.5px;">POOR</span>'
        '<span style="font-size:10px;font-weight:700;color:#9e9e9e !important;letter-spacing:0.5px;">AVERAGE</span>'
        '<span style="font-size:10px;font-weight:700;color:#be0000 !important;letter-spacing:0.5px;">GREAT</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    for label, val, pct, fmt, hib in metrics_data:
        color = savant_color(pct, hib)
        display_pct = int(round(pct)) if not pd.isna(pct) else "-"
        display_val = f"{val:{fmt}}" if not pd.isna(val) else "-"
        bar_left = max(min(pct, 100), 0) if not pd.isna(pct) else 50

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
            f'font-size:10px;font-weight:800;color:white !important;">{display_pct}</div>'
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
    return dict(type="rect", x0=-0.83, x1=0.83, y0=1.5, y1=3.5,
                line=dict(color="#333", width=2), fillcolor="rgba(0,0,0,0)")


def add_strike_zone(fig):
    fig.add_shape(strike_zone_rect())
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
def page_hitter_card(data):
    hitting = filter_davidson(data, "batter")
    if hitting.empty:
        st.warning("No hitting data found.")
        return

    batters = sorted(hitting["Batter"].unique())
    c1, c2 = st.columns([2, 3])
    with c1:
        selected = st.selectbox("Select Hitter", batters, key="hc_b")
    with c2:
        all_seasons = sorted(data["Season"].dropna().unique())
        sel_seasons = st.multiselect("Season", all_seasons, default=all_seasons, key="hc_s")

    all_stats = compute_batter_stats(data, season_filter=sel_seasons)
    if all_stats.empty or selected not in all_stats["Batter"].values:
        st.info("Not enough data for this player.")
        return

    pr = all_stats[all_stats["Batter"] == selected].iloc[0]
    bdf = hitting[(hitting["Batter"] == selected) & (hitting["Season"].isin(sel_seasons))]
    in_play = bdf[bdf["PitchCall"] == "InPlay"]
    batted = in_play.dropna(subset=["ExitSpeed"])

    jersey = JERSEY.get(selected, "")
    pos = POSITION.get(selected, "")
    side = bdf["BatterSide"].mode().iloc[0] if bdf["BatterSide"].notna().any() else ""
    bats = {"Right": "R", "Left": "L", "Switch": "S"}.get(side, side)

    player_header(selected, jersey, pos,
                  f"{pos}  |  Bats: {bats}  |  Davidson Wildcats",
                  f"{int(pr['PA'])} PA  |  {int(pr['BBE'])} Batted Balls  |  "
                  f"Seasons: {', '.join(str(int(s)) for s in sorted(sel_seasons))}")

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
        st.caption(f"vs. {len(all_stats)} batters in database (min 5 PA)")

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
            # Barrel zone
            fig.add_shape(type="rect", x0=8, x1=32, y0=98, y1=ev_la["ExitSpeed"].max() + 5,
                          line=dict(color="#e63946", width=1.5, dash="dash"),
                          fillcolor="rgba(230,57,70,0.06)")
            fig.add_annotation(x=20, y=ev_la["ExitSpeed"].max() + 3, text="BARREL ZONE",
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
        sub_pa = int(sub["PitchofPA"].eq(1).sum())
        sub_ks = len(sub[sub["KorBB"] == "Strikeout"])
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
                "chase": len(grp[(~((grp["PlateLocSide"].abs() <= 0.83) & grp["PlateLocHeight"].between(1.5, 3.5))) & grp["PlateLocSide"].notna()][grp["PitchCall"].isin(SWING_CALLS)]) / max(len(grp[~((grp["PlateLocSide"].abs() <= 0.83) & grp["PlateLocHeight"].between(1.5, 3.5)) & grp["PlateLocSide"].notna()]), 1) * 100 if grp["PlateLocSide"].notna().any() else np.nan,
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
    # Trackman: positive HB = arm-side run (toward 1B for RHP)
    # Savant chart: LEFT = toward 1B, RIGHT = toward 3B
    # So we negate HB so that arm-side (positive trackman) plots LEFT (negative x)
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
    fig.add_shape(type="rect", x0=-0.83, x1=0.83, y0=1.5, y1=3.5,
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
# PAGE: PITCHER CARD (Statcast Style)
# ──────────────────────────────────────────────
def page_pitcher_card(data):
    pitching = filter_davidson(data, "pitcher")
    if pitching.empty:
        st.warning("No pitching data found.")
        return

    pitchers = sorted(pitching["Pitcher"].unique())
    c1, c2 = st.columns([2, 3])
    with c1:
        selected = st.selectbox("Select Pitcher", pitchers, key="pc_p")
    with c2:
        all_seasons = sorted(data["Season"].dropna().unique())
        sel_seasons = st.multiselect("Season", all_seasons, default=all_seasons, key="pc_s")

    all_stats = compute_pitcher_stats(data, season_filter=sel_seasons)
    if all_stats.empty or selected not in all_stats["Pitcher"].values:
        st.info("Not enough data.")
        return

    pr = all_stats[all_stats["Pitcher"] == selected].iloc[0]
    pdf_raw = pitching[(pitching["Pitcher"] == selected) & (pitching["Season"].isin(sel_seasons))]
    # Filter to main pitches only (>= 5% usage)
    pdf = filter_minor_pitches(pdf_raw)
    if pdf.empty:
        st.info("Not enough pitch data after filtering.")
        return

    jersey = JERSEY.get(selected, "")
    pos = POSITION.get(selected, "")
    throws = pdf["PitcherThrows"].mode().iloc[0] if pdf["PitcherThrows"].notna().any() else ""
    thr = {"Right": "R", "Left": "L"}.get(throws, throws)

    # Arsenal summary line
    pitch_counts = pdf["TaggedPitchType"].value_counts()
    total_pitches = len(pdf)
    arsenal_summary = " | ".join(
        f"{pt} ({count / total_pitches * 100:.0f}%)" for pt, count in pitch_counts.items()
    )

    player_header(selected, jersey, pos,
                  f"{pos}  |  Throws: {thr}  |  Davidson Wildcats",
                  f"{total_pitches} pitches  |  {int(pr['PA'])} PA faced  |  "
                  f"Seasons: {', '.join(str(int(s)) for s in sorted(sel_seasons))}")

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
        st.caption(f"vs. {len(all_stats)} pitchers in database (min 20 pitches)")

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
            row["Tilt"] = sub["Tilt"].mode().iloc[0] if sub["Tilt"].notna().any() else None
        if "ZoneSpeed" in sub.columns and sub["ZoneSpeed"].notna().any():
            row["Zone Velo"] = round(sub["ZoneSpeed"].mean(), 1)
        if "VertApprAngle" in sub.columns and sub["VertApprAngle"].notna().any():
            row["VAA"] = round(sub["VertApprAngle"].mean(), 1)
        arsenal_rows.append(row)
    if arsenal_rows:
        st.dataframe(pd.DataFrame(arsenal_rows), use_container_width=True, hide_index=True)

    # Arsenal summary text
    st.markdown(
        f'<p style="font-size:13px;color:#555;margin-top:4px;">'
        f'{display_name(selected)} relies on {len(main_pitches)} pitches: {arsenal_summary}</p>',
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
            # Get all pitchers' whiff/ev vs this side
            all_side = data[data["BatterSide"] == side]
            pitcher_rows = []
            for pitcher, grp in all_side.groupby("Pitcher"):
                if len(grp) < 20:
                    continue
                sw = grp[grp["PitchCall"].isin(SWING_CALLS)]
                wh = grp[grp["PitchCall"] == "StrikeSwinging"]
                ip = grp[grp["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
                pitcher_rows.append({
                    "whiff": len(wh) / max(len(sw), 1) * 100 if len(sw) > 0 else np.nan,
                    "ev": ip["ExitSpeed"].mean() if len(ip) > 0 else np.nan,
                })
            all_df = pd.DataFrame(pitcher_rows)
            whiff_pct = get_percentile(p_whiff, all_df["whiff"]) if not all_df.empty and not pd.isna(p_whiff) else np.nan
            ev_pct = get_percentile(p_ev, all_df["ev"]) if not all_df.empty and not pd.isna(p_ev) else np.nan
            platoon_metrics.append((f"{label} Whiff%", p_whiff, whiff_pct, ".1f", True))
            if not pd.isna(ev_pct):
                platoon_metrics.append((f"{label} EV Against", p_ev, ev_pct, ".1f", False))

        if platoon_metrics:
            render_savant_percentile_section(platoon_metrics, None)
            st.caption("Percentile vs. all pitchers in DB (min 20 pitches vs that side)")

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
                # Context vs all pitchers in that set
                all_set = data[data["PitcherSet"] == s_val] if "PitcherSet" in data.columns else pd.DataFrame()
                pitcher_stats = []
                if not all_set.empty:
                    for p, grp in all_set.groupby("Pitcher"):
                        if len(grp) < 20:
                            continue
                        s_sw = grp[grp["PitchCall"].isin(SWING_CALLS)]
                        s_wh = grp[grp["PitchCall"] == "StrikeSwinging"]
                        s_fb = grp[grp["TaggedPitchType"].isin(["Fastball", "Sinker", "Cutter"])]
                        pitcher_stats.append({
                            "whiff": len(s_wh) / max(len(s_sw), 1) * 100 if len(s_sw) > 0 else np.nan,
                            "velo": s_fb["RelSpeed"].mean() if len(s_fb) > 0 else np.nan,
                        })
                all_df = pd.DataFrame(pitcher_stats)
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
                # Context: all pitchers' first pitch strike%
                all_fp = data[data["PitchofPA"] == 1]
                fp_pitcher_stats = []
                for p, grp in all_fp.groupby("Pitcher"):
                    if len(grp) < 10:
                        continue
                    stks = grp[grp["PitchCall"].isin(
                        ["StrikeCalled", "StrikeSwinging", "FoulBall", "FoulBallNotFieldable",
                         "FoulBallFieldable", "InPlay"])]
                    fp_pitcher_stats.append(len(stks) / max(len(grp), 1) * 100)
                fp_pct = get_percentile(fp_strike_pct, pd.Series(fp_pitcher_stats)) if fp_pitcher_stats else 50
                render_savant_percentile_section(
                    [("1st Pitch K%", fp_strike_pct, fp_pct, ".1f", True)], None,
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
        all_seasons = sorted(data["Season"].dropna().unique())
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
            in_zone = (loc_data["PlateLocSide"].abs() <= 0.83) & (loc_data["PlateLocHeight"].between(1.5, 3.5))
            out_zone_pitches = loc_data[~in_zone]
            called_strikes_out = out_zone_pitches[out_zone_pitches["PitchCall"] == "StrikeCalled"]
            frame_rate = len(called_strikes_out) / max(len(out_zone_pitches), 1) * 100

            # Context: all catchers' framing rate
            all_loc = data.dropna(subset=["PlateLocSide", "PlateLocHeight"])
            if "Catcher" in all_loc.columns:
                all_in_zone = (all_loc["PlateLocSide"].abs() <= 0.83) & (all_loc["PlateLocHeight"].between(1.5, 3.5))
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
        in_z = (loc["PlateLocSide"].abs() <= 0.83) & (loc["PlateLocHeight"].between(1.5, 3.5))
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
    st.header("Team Overview")
    all_seasons = sorted(data["Season"].dropna().unique())
    sel = st.multiselect("Season", all_seasons, default=all_seasons, key="to_s")

    tab_h, tab_p = st.tabs(["Hitting Leaderboard", "Pitching Leaderboard"])

    with tab_h:
        bs = compute_batter_stats(data, season_filter=sel)
        dav = bs[(bs["BatterTeam"] == DAVIDSON_TEAM_ID) & (bs["Batter"].isin(ROSTER_2026))].copy()
        if dav.empty:
            st.info("No hitting data.")
            return
        d = dav[["Batter", "PA", "BBE", "AvgEV", "MaxEV", "HardHitPct", "BarrelPct",
                 "SweetSpotPct", "WhiffPct", "KPct", "BBPct", "ChasePct"]].sort_values("PA", ascending=False).copy()
        d.columns = ["Batter", "PA", "BBE", "Avg EV", "Max EV", "Hard%", "Barrel%",
                     "Sweet%", "Whiff%", "K%", "BB%", "Chase%"]
        for c in d.columns[3:]:
            d[c] = d[c].round(1)
        st.dataframe(d, use_container_width=True, hide_index=True)

    with tab_p:
        ps = compute_pitcher_stats(data, season_filter=sel)
        dav = ps[(ps["PitcherTeam"] == DAVIDSON_TEAM_ID) & (ps["Pitcher"].isin(ROSTER_2026))].copy()
        if dav.empty:
            st.info("No pitching data.")
            return
        d = dav[["Pitcher", "Pitches", "PA", "AvgFBVelo", "MaxFBVelo", "AvgSpin",
                 "WhiffPct", "KPct", "BBPct", "ZonePct", "ChasePct", "AvgEVAgainst", "GBPct"]].sort_values("Pitches", ascending=False).copy()
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
# PAGE: GAME LOG
# ──────────────────────────────────────────────
def page_game_log(data):
    st.header("Game Log")
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
        sel = st.selectbox("Drill into game", opts,
                           format_func=lambda g: f"{games[games['GameID']==g]['Date'].iloc[0].strftime('%Y-%m-%d')} "
                                                 f"{games[games['GameID']==g]['Away'].iloc[0]} @ "
                                                 f"{games[games['GameID']==g]['Home'].iloc[0]}")
        gd = dav[dav["GameID"] == sel]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Davidson Pitching**")
            gp = gd[gd["PitcherTeam"] == DAVIDSON_TEAM_ID]
            for p in gp["Pitcher"].unique():
                sub = gp[gp["Pitcher"] == p]
                st.markdown(f"_{p}_ - {len(sub)} pitches")
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
                st.markdown(f"_{b}_ - {len(sub)} BIP, Avg: {sub['ExitSpeed'].mean():.1f}, "
                            f"Max: {sub['ExitSpeed'].max():.1f}")


# ──────────────────────────────────────────────
# PAGE: SCOUTING
# ──────────────────────────────────────────────
def page_scouting(data):
    st.header("Opponent Scouting")
    dav = data[(data["PitcherTeam"] == DAVIDSON_TEAM_ID) | (data["BatterTeam"] == DAVIDSON_TEAM_ID)]
    teams = set()
    for c in ["PitcherTeam", "BatterTeam"]:
        teams.update(dav[c].dropna().unique())
    teams.discard(DAVIDSON_TEAM_ID)
    teams = sorted(teams)
    if not teams:
        st.info("No opponent data.")
        return

    team = st.selectbox("Opponent", teams)
    role = st.radio("View", ["Their Pitching", "Their Hitting"], horizontal=True, key="sc_r")

    if role == "Their Pitching":
        opp = data[data["PitcherTeam"] == team]
        if opp.empty:
            st.info("No data.")
            return
        p = st.selectbox("Pitcher", sorted(opp["Pitcher"].unique()), key="sc_p")
        sub = opp[opp["Pitcher"] == p]
        rows = []
        for pt in sorted(sub["TaggedPitchType"].dropna().unique()):
            s = sub[sub["TaggedPitchType"] == pt]
            if len(s) < 2:
                continue
            rows.append({"Pitch": pt, "N": len(s),
                         "Velo": round(s["RelSpeed"].mean(), 1) if s["RelSpeed"].notna().any() else None,
                         "Spin": int(round(s["SpinRate"].mean())) if s["SpinRate"].notna().any() else None,
                         "IVB": round(s["InducedVertBreak"].mean(), 1) if s["InducedVertBreak"].notna().any() else None,
                         "HB": round(s["HorzBreak"].mean(), 1) if s["HorzBreak"].notna().any() else None})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        loc = sub.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        if not loc.empty:
            fig = px.density_heatmap(loc, x="PlateLocSide", y="PlateLocHeight", nbinsx=12, nbinsy=12,
                                     color_continuous_scale="YlOrRd")
            add_strike_zone(fig)
            fig.update_layout(xaxis=dict(range=[-3, 3], scaleanchor="y"),
                              yaxis=dict(range=[0, 5]),
                              height=400, coloraxis_showscale=False, **CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
    else:
        opp = data[data["BatterTeam"] == team]
        if opp.empty:
            st.info("No data.")
            return
        b = st.selectbox("Batter", sorted(opp["Batter"].unique()), key="sc_b")
        sub = opp[opp["Batter"] == b]
        batted = sub[sub["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Avg EV", f"{batted['ExitSpeed'].mean():.1f}" if len(batted) > 0 else "-")
        with c2:
            st.metric("Max EV", f"{batted['ExitSpeed'].max():.1f}" if len(batted) > 0 else "-")
        with c3:
            sw = sub[sub["PitchCall"].isin(SWING_CALLS)]
            wh = sub[sub["PitchCall"] == "StrikeSwinging"]
            st.metric("Whiff%", f"{len(wh)/max(len(sw),1)*100:.1f}%" if len(sw) > 0 else "-")
        loc = sub.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        sl = loc[loc["PitchCall"].isin(SWING_CALLS)]
        if not sl.empty:
            fig = px.density_heatmap(sl, x="PlateLocSide", y="PlateLocHeight", nbinsx=12, nbinsy=12,
                                     color_continuous_scale="YlOrRd")
            add_strike_zone(fig)
            fig.update_layout(xaxis=dict(range=[-3, 3], scaleanchor="y"),
                              yaxis=dict(range=[0, 5]),
                              height=400, coloraxis_showscale=False, **CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# PITCH DESIGN LAB
# ──────────────────────────────────────────────

def _compute_stuff_plus(data, baseline=None):
    """Compute Stuff+ for every pitch in data.
    Model: z-score composite of velo, IVB, HB, extension, VAA, spin rate
    relative to same pitch type across the BASELINE population.
    100 = average, each 10 = 1 stdev better.

    Args:
        data: DataFrame of pitches to score
        baseline: DataFrame to compute mean/std from (full team data).
                  If None, uses data itself (should be full team).
    """
    df = data.dropna(subset=["RelSpeed", "TaggedPitchType"]).copy()
    if df.empty:
        return df

    if baseline is None:
        baseline = df
    base = baseline.dropna(subset=["RelSpeed", "TaggedPitchType"]).copy()

    # Pitch-type-specific weights: what matters most for each archetype
    weights = {
        "Fastball":  {"RelSpeed": 2.0, "InducedVertBreak": 2.0, "HorzBreak": 0.5, "Extension": 1.0, "VertApprAngle": 1.5, "SpinRate": 0.8},
        "Sinker":    {"RelSpeed": 1.5, "InducedVertBreak": -1.0, "HorzBreak": 1.5, "Extension": 1.0, "VertApprAngle": -1.0, "SpinRate": 0.5},
        "Cutter":    {"RelSpeed": 1.5, "InducedVertBreak": 0.5, "HorzBreak": 1.5, "Extension": 1.0, "VertApprAngle": 1.0, "SpinRate": 0.8},
        "Slider":    {"RelSpeed": 0.8, "InducedVertBreak": -1.5, "HorzBreak": 2.0, "Extension": 0.8, "VertApprAngle": -1.0, "SpinRate": 1.0},
        "Curveball": {"RelSpeed": -0.5, "InducedVertBreak": -2.0, "HorzBreak": 1.5, "Extension": 0.5, "VertApprAngle": -1.5, "SpinRate": 1.5},
        "Changeup":  {"RelSpeed": -1.5, "InducedVertBreak": -1.0, "HorzBreak": 1.5, "Extension": 1.0, "VertApprAngle": -1.0, "SpinRate": -0.5},
        "Sweeper":   {"RelSpeed": 0.5, "InducedVertBreak": -1.5, "HorzBreak": 2.5, "Extension": 0.8, "VertApprAngle": -1.0, "SpinRate": 1.0},
        "Splitter":  {"RelSpeed": 1.0, "InducedVertBreak": -2.0, "HorzBreak": 0.5, "Extension": 1.2, "VertApprAngle": -1.5, "SpinRate": 0.3},
        "Knuckle Curve": {"RelSpeed": -0.5, "InducedVertBreak": -2.0, "HorzBreak": 1.5, "Extension": 0.5, "VertApprAngle": -1.5, "SpinRate": 1.5},
    }
    default_w = {"RelSpeed": 1.0, "InducedVertBreak": 1.0, "HorzBreak": 1.0, "Extension": 1.0, "VertApprAngle": 1.0, "SpinRate": 1.0}

    # Pre-compute baseline stats per pitch type
    baseline_stats = {}
    for pt, bgrp in base.groupby("TaggedPitchType"):
        stats = {}
        for col in ["RelSpeed", "InducedVertBreak", "HorzBreak", "Extension", "VertApprAngle", "SpinRate"]:
            if col in bgrp.columns:
                vals = bgrp[col].astype(float).dropna()
                stats[col] = (vals.mean(), vals.std())
        baseline_stats[pt] = stats

    stuff_scores = []
    for pt, grp in df.groupby("TaggedPitchType"):
        w = weights.get(pt, default_w)
        bstats = baseline_stats.get(pt, {})
        z_total = pd.Series(0.0, index=grp.index)
        w_total = 0.0
        for col, weight in w.items():
            if col not in grp.columns or col not in bstats:
                continue
            mu, sigma = bstats[col]
            if sigma == 0 or pd.isna(sigma) or pd.isna(mu):
                continue
            vals = grp[col].astype(float)
            z = (vals - mu) / sigma
            z_total += z * weight
            w_total += abs(weight)
        if w_total > 0:
            z_total = z_total / w_total
        grp = grp.copy()
        grp["StuffPlus"] = 100 + z_total * 10
        stuff_scores.append(grp)

    if stuff_scores:
        return pd.concat(stuff_scores, ignore_index=True)
    return df


def _compute_tunnel_score(pdf):
    """Compute tunnel scores using physics-based commit-point analysis.
    True tunneling = pitches are indistinguishable at the hitter's commit point
    (~167ms before plate) but diverge significantly by the time they arrive.
    Score uses actual flight path modeling, not just averages."""
    req = ["TaggedPitchType", "RelHeight", "RelSide", "PlateLocHeight", "PlateLocSide",
           "InducedVertBreak", "HorzBreak", "RelSpeed"]
    if not all(c in pdf.columns for c in req):
        return pd.DataFrame()

    pitch_types = pdf["TaggedPitchType"].unique()
    if len(pitch_types) < 2:
        return pd.DataFrame()

    MOUND_DIST = 60.5

    # Compute per-pitch-type averages
    agg_cols = {
        "rel_h": ("RelHeight", "mean"), "rel_s": ("RelSide", "mean"),
        "rel_h_std": ("RelHeight", "std"), "rel_s_std": ("RelSide", "std"),
        "loc_h": ("PlateLocHeight", "mean"), "loc_s": ("PlateLocSide", "mean"),
        "ivb": ("InducedVertBreak", "mean"), "hb": ("HorzBreak", "mean"),
        "velo": ("RelSpeed", "mean"), "count": ("RelSpeed", "count"),
    }
    if "Extension" in pdf.columns:
        agg_cols["ext"] = ("Extension", "mean")
    agg = pdf.groupby("TaggedPitchType").agg(**agg_cols).dropna(subset=["rel_h", "velo"])
    if "ext" not in agg.columns:
        agg["ext"] = 6.0

    def _flight_pos_at_frac(row, frac):
        """Get (x, y) position at a fraction of flight path (0=release, 1=plate)."""
        ext = row.ext if not pd.isna(row.ext) else 6.0
        actual_dist = MOUND_DIST - ext
        velo_fps = row.velo * 5280 / 3600
        t_total = actual_dist / velo_fps
        t = frac * t_total
        gravity_drop = 0.5 * 32.17 * t**2
        ivb_lift = (row.ivb / 12.0) * frac**2
        y = row.rel_h + (row.loc_h - row.rel_h) * frac - gravity_drop + ivb_lift
        y_correction = row.loc_h - (row.rel_h + (row.loc_h - row.rel_h) - 0.5 * 32.17 * t_total**2 + (row.ivb / 12.0))
        y = row.rel_h + (row.loc_h - row.rel_h) * frac - gravity_drop + ivb_lift
        # Apply endpoint correction
        y_at_1 = row.rel_h + (row.loc_h - row.rel_h) - 0.5 * 32.17 * t_total**2 + (row.ivb / 12.0)
        y += (row.loc_h - y_at_1) * frac

        hb_curve = (row.hb / 12.0) * frac**2
        x = row.rel_s + (row.loc_s - row.rel_s) * frac + hb_curve
        x_at_1 = row.rel_s + (row.loc_s - row.rel_s) + (row.hb / 12.0)
        x += (row.loc_s - x_at_1) * frac
        return x, y

    rows = []
    types = list(agg.index)
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            a, b = agg.loc[types[i]], agg.loc[types[j]]

            # Release point separation (inches)
            rel_sep = np.sqrt((a.rel_h - b.rel_h)**2 + (a.rel_s - b.rel_s)**2) * 12

            # Commit point separation (at 60% of flight ≈ hitter's decision point)
            ax, ay = _flight_pos_at_frac(a, 0.6)
            bx, by = _flight_pos_at_frac(b, 0.6)
            commit_sep = np.sqrt((ay - by)**2 + (ax - bx)**2) * 12  # inches

            # Plate separation
            plate_sep = np.sqrt((a.loc_h - b.loc_h)**2 + (a.loc_s - b.loc_s)**2) * 12

            # Movement divergence
            move_div = np.sqrt((a.ivb - b.ivb)**2 + (a.hb - b.hb)**2)
            velo_gap = abs(a.velo - b.velo)

            # TUNNEL SCORE: low commit-point separation + high plate separation = good
            # Penalize: high release separation, high commit separation
            # Reward: high plate separation, high movement divergence
            if commit_sep < 0.1:
                commit_sep = 0.1  # avoid division by zero
            divergence_ratio = plate_sep / commit_sep  # higher = better tunnel

            # Release consistency penalty (inconsistent release = hitter can read it early)
            rel_penalty = max(0, 1 - rel_sep / 6.0)  # >6 inches apart at release = 0

            # Commit-point deception (under 2" = elite, over 6" = bad)
            commit_deception = max(0, 1 - commit_sep / 8.0)

            # Plate divergence reward (more is better, but cap at reasonable range)
            plate_reward = min(plate_sep / 12.0, 1.5)  # normalize: 12" = 1.0, cap at 1.5

            # Combined tunnel score (0-100 scale with real differentiation)
            raw = (commit_deception * 0.45 + rel_penalty * 0.20 +
                   min(divergence_ratio / 3.0, 1.0) * 0.20 +
                   min(plate_reward, 1.0) * 0.15) * 100
            tunnel = round(min(raw, 100), 1)

            # Letter grade with strict thresholds
            if tunnel >= 75:
                grade = "A"
            elif tunnel >= 60:
                grade = "B"
            elif tunnel >= 45:
                grade = "C"
            elif tunnel >= 30:
                grade = "D"
            else:
                grade = "F"

            # Actionable diagnosis
            issues = []
            fixes = []
            if rel_sep > 4:
                issues.append(f"release points {rel_sep:.0f}\" apart")
                fixes.append("Work on consistent arm slot across both pitches")
            if commit_sep > 5:
                issues.append(f"{commit_sep:.0f}\" apart at commit point")
                if velo_gap > 8:
                    fixes.append(f"Reduce {velo_gap:.0f} mph velo gap — pitches separate too early")
                else:
                    fixes.append("Pitch trajectories diverge too early — hitter can read them")
            if plate_sep < 6:
                issues.append(f"only {plate_sep:.0f}\" apart at plate")
                fixes.append("Pitches end up too close together — need more movement contrast")
            if move_div < 5:
                issues.append(f"only {move_div:.0f}\" movement difference")
                fixes.append("Increase break differential — pitches move too similarly")
            if not issues:
                if tunnel >= 60:
                    diagnosis = "Strong tunnel — pitches look identical until it's too late"
                else:
                    diagnosis = "Decent pairing but room to tighten"
            else:
                diagnosis = "; ".join(issues)

            rows.append({
                "Pitch A": types[i], "Pitch B": types[j],
                "Grade": grade, "Tunnel Score": tunnel,
                "Release Sep (in)": round(rel_sep, 1),
                "Commit Sep (in)": round(commit_sep, 1),
                "Plate Sep (in)": round(plate_sep, 1),
                "Velo Gap (mph)": round(velo_gap, 1),
                "Move Diff (in)": round(move_div, 1),
                "Diagnosis": diagnosis,
                "Fix": "; ".join(fixes) if fixes else "No changes needed",
            })
    return pd.DataFrame(rows).sort_values("Tunnel Score", ascending=False).reset_index(drop=True)


def _compute_pitch_pair_results(pdf, data):
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
        if n < 5:
            continue
        swings = grp[is_swing.reindex(grp.index, fill_value=False)]
        whiffs = grp[is_whiff.reindex(grp.index, fill_value=False)]
        csws = grp[is_csw.reindex(grp.index, fill_value=False)]
        batted = grp[(grp["PitchCall"] == "InPlay") & grp["ExitSpeed"].notna()]
        rows.append({
            "Setup Pitch": prev, "Follow Pitch": curr, "Count": n,
            "Whiff%": round(len(whiffs) / max(len(swings), 1) * 100, 1),
            "CSW%": round(len(csws) / n * 100, 1),
            "Avg EV": round(batted["ExitSpeed"].mean(), 1) if len(batted) > 0 else np.nan,
            "Chase%": round(
                len(grp[(~((grp["PlateLocSide"].abs() <= 0.83) & grp["PlateLocHeight"].between(1.5, 3.5))) &
                         grp["PitchCall"].isin(SWING_CALLS)]) /
                max(len(grp[~((grp["PlateLocSide"].abs() <= 0.83) & grp["PlateLocHeight"].between(1.5, 3.5))]), 1) * 100, 1),
        })
    return pd.DataFrame(rows).sort_values("Whiff%", ascending=False).reset_index(drop=True)


def _generate_ai_report(pdf, pitcher_name, stuff_df, tunnel_df, pair_df, all_data):
    """Generate comprehensive AI scouting report with actionable recommendations."""
    lines = []
    lines.append(f"## AI Pitch Design Report: {display_name(pitcher_name)}")
    lines.append("")

    if stuff_df.empty:
        lines.append("*Insufficient data to generate report.*")
        return "\n".join(lines)

    # ── Arsenal Overview ──
    arsenal = stuff_df.groupby("TaggedPitchType").agg(
        count=("StuffPlus", "count"),
        stuff_avg=("StuffPlus", "mean"),
        stuff_std=("StuffPlus", "std"),
        velo=("RelSpeed", "mean"),
        ivb=("InducedVertBreak", "mean"),
        hb=("HorzBreak", "mean"),
        spin=("SpinRate", "mean"),
        vaa=("VertApprAngle", "mean"),
        ext=("Extension", "mean"),
    ).sort_values("count", ascending=False)

    best_pitch = arsenal["stuff_avg"].idxmax()
    best_stuff = arsenal.loc[best_pitch, "stuff_avg"]
    worst_pitch = arsenal["stuff_avg"].idxmin()
    worst_stuff = arsenal.loc[worst_pitch, "stuff_avg"]

    lines.append("### Arsenal Grades")
    lines.append("")
    for pt in arsenal.index:
        s = arsenal.loc[pt, "stuff_avg"]
        grade = "Elite" if s >= 120 else "Plus" if s >= 110 else "Above Avg" if s >= 105 else "Average" if s >= 95 else "Below Avg" if s >= 90 else "Poor"
        emoji = "A+" if s >= 120 else "A" if s >= 110 else "B+" if s >= 105 else "B" if s >= 95 else "C" if s >= 90 else "D"
        count = int(arsenal.loc[pt, "count"])
        velo = arsenal.loc[pt, "velo"]
        lines.append(f"- **{pt}** ({emoji}): Stuff+ {s:.0f} | {velo:.1f} mph | {count} pitches | *{grade}*")
    lines.append("")

    # ── Strengths ──
    lines.append("### Key Strengths")
    lines.append("")
    if best_stuff >= 105:
        lines.append(f"- **{best_pitch}** is the clear weapon pitch (Stuff+ {best_stuff:.0f})")
        bv = arsenal.loc[best_pitch, "velo"]
        bi = arsenal.loc[best_pitch, "ivb"]
        bh = arsenal.loc[best_pitch, "hb"]
        if best_pitch in ("Fastball", "Sinker", "Cutter"):
            if bi > 15:
                lines.append(f"  - Elite vertical carry ({bi:.1f} in IVB) — hitters swing under this pitch")
            if bv >= 90:
                lines.append(f"  - Plus velocity ({bv:.1f} mph) creates swing-and-miss at the top of the zone")
        else:
            if abs(bh) > 10:
                lines.append(f"  - Elite horizontal movement ({bh:.1f} in) generates chase swings")
            if bi < -5:
                lines.append(f"  - Strong vertical drop ({bi:.1f} in IVB) plays well below the zone")

    # Release consistency
    rel_std_h = pdf.groupby("TaggedPitchType")["RelHeight"].std().mean()
    rel_std_s = pdf.groupby("TaggedPitchType")["RelSide"].std().mean()
    avg_rel_std = (rel_std_h + rel_std_s) / 2 if not pd.isna(rel_std_h) else 1.0
    if avg_rel_std < 0.15:
        lines.append("- **Excellent release point consistency** — all pitches look the same out of the hand")
    elif avg_rel_std < 0.25:
        lines.append("- **Good release point consistency** — pitches tunnel well at the release point")

    # Velo spread
    fb_types = [t for t in arsenal.index if t in ("Fastball", "Sinker", "Cutter")]
    off_types = [t for t in arsenal.index if t in ("Changeup", "Splitter")]
    if fb_types and off_types:
        fb_velo = arsenal.loc[fb_types, "velo"].max()
        off_velo = arsenal.loc[off_types, "velo"].min()
        velo_diff = fb_velo - off_velo
        if velo_diff >= 10:
            lines.append(f"- **Strong velocity differential** ({velo_diff:.0f} mph gap) — effective speed change disrupts timing")

    lines.append("")

    # ── Areas for Improvement ──
    lines.append("### Areas for Improvement")
    lines.append("")
    if worst_stuff < 95:
        lines.append(f"- **{worst_pitch}** needs work (Stuff+ {worst_stuff:.0f})")
        wv = arsenal.loc[worst_pitch, "velo"]
        wi = arsenal.loc[worst_pitch, "ivb"]
        wh = arsenal.loc[worst_pitch, "hb"]
        if worst_pitch in ("Slider", "Sweeper", "Curveball", "Knuckle Curve") and abs(wh) < 5:
            lines.append(f"  - *Recommendation*: Increase horizontal break — currently only {wh:.1f} in, aim for 8+ in")
        if worst_pitch == "Changeup" and fb_types:
            fb_v = arsenal.loc[fb_types[0], "velo"]
            diff = fb_v - wv
            if diff < 8:
                lines.append(f"  - *Recommendation*: Increase velo separation — only {diff:.1f} mph gap (want 8-12 mph)")
        if worst_pitch in ("Fastball", "Sinker") and wi < 12:
            lines.append(f"  - *Recommendation*: Increase vertical carry — {wi:.1f} in IVB is below average (target 14+)")

    if avg_rel_std >= 0.25:
        lines.append(f"- **Release point inconsistency** (avg deviation: {avg_rel_std:.2f} ft) — work on repeating delivery")

    # Check for missing pitch types
    has_fb = any(t in arsenal.index for t in ("Fastball", "Sinker"))
    has_breaking = any(t in arsenal.index for t in ("Slider", "Curveball", "Sweeper", "Knuckle Curve"))
    has_offspeed = any(t in arsenal.index for t in ("Changeup", "Splitter"))
    if has_fb and not has_offspeed:
        lines.append("- **No offspeed pitch in arsenal** — adding a changeup would give hitters a different look and speed change")
    if has_fb and not has_breaking:
        lines.append("- **No breaking ball in arsenal** — adding a slider or curve would attack a different plane of movement")

    lines.append("")

    # ── Tunnel Recommendations ──
    if not tunnel_df.empty:
        lines.append("### Pitch Tunnel Analysis")
        lines.append("")
        for _, r in tunnel_df.iterrows():
            grade = r.get("Grade", "?")
            score = r.get("Tunnel Score", 0)
            diag = r.get("Diagnosis", "")
            fix = r.get("Fix", "")
            lines.append(f"- **{r['Pitch A']} + {r['Pitch B']}** — Grade: **{grade}** (Score: {score})")
            lines.append(f"  - Release: {r['Release Sep (in)']}\" | Commit: {r['Commit Sep (in)']}\" | Plate: {r['Plate Sep (in)']}\"")
            lines.append(f"  - *{diag}*")
            if fix and fix != "No changes needed":
                lines.append(f"  - **Action:** {fix}")
        lines.append("")

    # ── Sequencing Recommendations ──
    if not pair_df.empty:
        lines.append("### Optimal Pitch Sequences")
        lines.append("")
        top_seq = pair_df[pair_df["Count"] >= 8].head(5)
        for _, r in top_seq.iterrows():
            ev_str = f" | EV Against: {r['Avg EV']:.1f}" if not pd.isna(r.get("Avg EV")) else ""
            lines.append(f"- **{r['Setup Pitch']} --> {r['Follow Pitch']}**: {r['Whiff%']:.0f}% Whiff, {r['CSW%']:.0f}% CSW{ev_str}")
        if len(top_seq) > 0:
            best_seq = top_seq.iloc[0]
            lines.append(f"\n  *Top combo*: Use **{best_seq['Setup Pitch']}** to set up **{best_seq['Follow Pitch']}** for swing-and-miss")
        lines.append("")

    # ── Location Strategy ──
    lines.append("### Location Strategy Insights")
    lines.append("")
    for pt in arsenal.index:
        pt_data = stuff_df[stuff_df["TaggedPitchType"] == pt]
        whiff_data = pt_data[pt_data["PitchCall"] == "StrikeSwinging"]
        if len(whiff_data) > 5:
            avg_h = whiff_data["PlateLocHeight"].mean()
            avg_s = whiff_data["PlateLocSide"].mean()
            zone_desc = []
            if avg_h > 3.0:
                zone_desc.append("elevated")
            elif avg_h < 2.0:
                zone_desc.append("low")
            if avg_s > 0.3:
                zone_desc.append("arm-side")
            elif avg_s < -0.3:
                zone_desc.append("glove-side")
            if zone_desc:
                loc_str = " and ".join(zone_desc)
                lines.append(f"- **{pt}** gets most whiffs when located **{loc_str}** (avg whiff loc: {avg_h:.1f}H, {avg_s:.1f}S)")
    lines.append("")

    # ── Overall Game Plan ──
    lines.append("### Recommended Game Plan")
    lines.append("")
    primary = best_pitch
    # Find best secondary
    secondary = None
    if not tunnel_df.empty:
        top_t = tunnel_df.iloc[0]
        secondary = top_t["Pitch B"] if top_t["Pitch A"] == primary else top_t["Pitch A"]
    elif len(arsenal) > 1:
        secondary = arsenal.index[1] if arsenal.index[0] == primary else arsenal.index[0]

    lines.append(f"1. **Establish {primary}** early in counts to set the tone")
    if secondary:
        lines.append(f"2. **Tunnel {primary} into {secondary}** — these two pitches pair well together")
        if not pair_df.empty:
            best_2strike = pair_df[(pair_df["Whiff%"] >= 25) & (pair_df["Count"] >= 5)]
            if len(best_2strike) > 0:
                bs = best_2strike.iloc[0]
                lines.append(f"3. **Putaway combo**: {bs['Setup Pitch']} --> {bs['Follow Pitch']} ({bs['Whiff%']:.0f}% whiff rate)")
    if has_offspeed:
        lines.append(f"4. **Use offspeed** to disrupt timing — especially effective after back-to-back fastballs")
    lines.append("")

    # ── Comparison to Database ──
    lines.append("### Database Percentile Rankings")
    lines.append("")
    # Compare this pitcher's stuff+ to all pitchers in the database
    all_pitchers_stuff = _compute_stuff_plus(all_data, baseline=all_data)
    if not all_pitchers_stuff.empty:
        for pt in arsenal.index:
            my_stuff = arsenal.loc[pt, "stuff_avg"]
            all_pt = all_pitchers_stuff[all_pitchers_stuff["TaggedPitchType"] == pt]["StuffPlus"]
            if len(all_pt) > 10:
                pctl = percentileofscore(all_pt.dropna(), my_stuff, kind="rank")
                lines.append(f"- **{pt}**: {pctl:.0f}th percentile Stuff+ across all pitchers in database")

    return "\n".join(lines)


def page_pitch_design_lab(data):
    st.markdown('<div class="section-header">Pitch Design Lab</div>', unsafe_allow_html=True)
    st.caption("AI-powered pitch analysis: Stuff+ grades, tunnel scores, sequencing optimization, and actionable recommendations")

    dav_pitching = filter_davidson(data, role="pitcher")
    if dav_pitching.empty:
        st.warning("No Davidson pitching data found.")
        return

    pitchers = sorted(dav_pitching["Pitcher"].unique())
    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        pitcher = st.selectbox("Select Pitcher", pitchers,
                               format_func=display_name, key="pdl_pitcher")
    with col_sel2:
        seasons = sorted(dav_pitching["Season"].dropna().unique())
        season_filter = st.multiselect("Season", seasons, default=seasons, key="pdl_season")

    pdf = dav_pitching[dav_pitching["Pitcher"] == pitcher].copy()
    if season_filter:
        pdf = pdf[pdf["Season"].isin(season_filter)]
    pdf = filter_minor_pitches(pdf)

    if len(pdf) < 20:
        st.warning("Not enough pitches (need 20+) to analyze.")
        return

    jersey = JERSEY.get(pitcher, "")
    pos = POSITION.get(pitcher, "P")
    player_header(pitcher, jersey, pos,
                  f"{len(pdf)} pitches analyzed | Seasons: {', '.join(str(s) for s in sorted(pdf['Season'].dropna().unique()))}",
                  "Pitch Design Lab")

    # Compute Stuff+
    stuff_df = _compute_stuff_plus(pdf, baseline=data)
    if "StuffPlus" not in stuff_df.columns:
        st.error("Could not compute Stuff+ scores.")
        return

    # ─── TAB LAYOUT ───
    tab_stuff, tab_tunnel, tab_seq, tab_loc, tab_sim, tab_cmd, tab_ai = st.tabs([
        "Stuff+ Grades", "Pitch Tunnels", "Sequencing", "Location Lab",
        "Hitter's Eye", "Command+", "AI Report"
    ])

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
        all_stuff = _compute_stuff_plus(data, baseline=data)
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
        st.caption("Tunnel Score measures how well two pitches look identical at the hitter's commit point but diverge at the plate. "
                   "Grades: A (elite deception) → F (hitter can read pitches early). Based on physics-modeled flight paths.")

        tunnel_df = _compute_tunnel_score(pdf)
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
                            "Velo Gap (mph)", "Move Diff (in)"]
            st.dataframe(tunnel_df[display_cols], use_container_width=True, hide_index=True)

            # Diagnosis & Fix cards for each pair
            section_header("Diagnosis & Recommendations")
            for _, row in tunnel_df.iterrows():
                gc = grade_colors.get(row["Grade"], "#888")
                border_color = gc
                st.markdown(
                    f'<div style="border-left:5px solid {border_color};padding:12px 16px;'
                    f'border-radius:4px;margin:8px 0;background:{gc}10;">'
                    f'<span style="font-size:18px;font-weight:bold;color:{gc};">{row["Grade"]}</span> '
                    f'<b>{row["Pitch A"]} + {row["Pitch B"]}</b> '
                    f'<span style="color:#666;">(Score: {row["Tunnel Score"]})</span><br>'
                    f'<span style="font-size:13px;"><b>Analysis:</b> {row["Diagnosis"]}</span><br>'
                    f'<span style="color:{gc};font-size:13px;"><b>Action:</b> {row["Fix"]}</span>'
                    f'</div>', unsafe_allow_html=True)

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
                        xaxis=dict(range=[-2.5, 2.5]), yaxis=dict(range=[0, 5]),
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

        pair_df = _compute_pitch_pair_results(pdf, data)
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

                    # AI insight for location
                    if not zone_df.empty:
                        best_zone = zone_df.loc[zone_df["Whiff%"].idxmax()]
                        worst_zone = zone_df.dropna(subset=["Avg EV"])
                        if not worst_zone.empty:
                            worst_zone = worst_zone.loc[worst_zone["Avg EV"].idxmax()]
                            st.markdown(
                                f'<div style="background:#f0f7ff;border-left:4px solid #2d7fc1;padding:12px 16px;'
                                f'border-radius:4px;margin:8px 0;">'
                                f'<b style="color:#1a1a2e;">AI Insight:</b><br>'
                                f'<span style="color:#333;">Best location for whiffs: <b>{best_zone["Zone"]}</b> '
                                f'({best_zone["Whiff%"]:.0f}% whiff rate). '
                                f'Avoid <b>{worst_zone["Zone"]}</b> — hitters average '
                                f'{worst_zone["Avg EV"]:.0f} mph exit velo there.</span></div>',
                                unsafe_allow_html=True,
                            )

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
                        type="rect", x0=-0.83, x1=0.83, y0=1.5, y1=3.5,
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
                                   zeroline=False, showgrid=True, gridcolor="#eee"),
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
                    st.caption("At the commit point (~167ms before plate), how far apart are the pitches? "
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

        # Compute Command+ for each pitch type
        cmd_rows = []
        for pt in sorted(pdf["TaggedPitchType"].unique()):
            ptd = pdf[pdf["TaggedPitchType"] == pt].dropna(subset=["PlateLocSide", "PlateLocHeight"])
            if len(ptd) < 10:
                continue

            # Location consistency (lower std = better command)
            loc_std_h = ptd["PlateLocHeight"].std()
            loc_std_s = ptd["PlateLocSide"].std()
            loc_spread = np.sqrt(loc_std_h**2 + loc_std_s**2)

            # Zone rate
            in_zone = ((ptd["PlateLocSide"].abs() <= 0.83) &
                       ptd["PlateLocHeight"].between(1.5, 3.5))
            zone_pct = in_zone.mean() * 100

            # Edge rate (borderline pitches — the best location)
            edge = (
                ((ptd["PlateLocSide"].abs().between(0.5, 1.1)) |
                 (ptd["PlateLocHeight"].between(1.2, 1.8)) |
                 (ptd["PlateLocHeight"].between(3.2, 3.8))) &
                (ptd["PlateLocSide"].abs() <= 1.5) &
                ptd["PlateLocHeight"].between(0.5, 4.5)
            )
            edge_pct = edge.mean() * 100

            # Called strike + whiff rate (CSW) — outcome measure of command
            csw = ptd["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"]).mean() * 100

            # Chase rate (ability to locate out of zone and get swings)
            out_zone = ~in_zone
            chase_swings = ptd[out_zone & ptd["PitchCall"].isin(SWING_CALLS)]
            chase_pct = len(chase_swings) / max(out_zone.sum(), 1) * 100

            # Command+ composite: normalized vs same pitch type across all Davidson pitchers
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
            st.info("Not enough location data to compute Command+.")
        else:
            cmd_df = pd.DataFrame(cmd_rows)

            # Compute Command+ score
            # Compare to all Davidson pitchers for the same pitch type
            all_dav = filter_davidson(data, role="pitcher")
            all_dav = normalize_pitch_types(all_dav)
            cmd_scores = []
            for _, row in cmd_df.iterrows():
                pt = row["Pitch"]
                all_pt = all_dav[all_dav["TaggedPitchType"] == pt].dropna(
                    subset=["PlateLocSide", "PlateLocHeight"])
                if len(all_pt) < 20:
                    cmd_scores.append(100.0)
                    continue
                # Per-pitcher spread for this pitch type
                pitcher_spreads = []
                for p, pg in all_pt.groupby("Pitcher"):
                    if len(pg) < 10:
                        continue
                    sp = np.sqrt(pg["PlateLocHeight"].std()**2 + pg["PlateLocSide"].std()**2)
                    pitcher_spreads.append(sp)
                if len(pitcher_spreads) < 3:
                    cmd_scores.append(100.0)
                    continue
                # Lower spread = better = higher Command+
                pctl = 100 - percentileofscore(pitcher_spreads, row["Loc Spread (ft)"], kind="rank")
                cmd_scores.append(round(100 + (pctl - 50) * 0.4, 0))

            cmd_df["Command+"] = cmd_scores
            cmd_df = cmd_df.sort_values("Command+", ascending=False)

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
                    xaxis=dict(range=[-2.5, 2.5], title="Plate Side (ft)"),
                    yaxis=dict(range=[0, 5.5], title="Plate Height (ft)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    **CHART_LAYOUT,
                )
                st.plotly_chart(fig_loc, use_container_width=True)

                # AI insight for command
                spread = np.sqrt(std_s**2 + std_h**2)
                grade = "Elite" if spread < 0.35 else "Plus" if spread < 0.45 else "Average" if spread < 0.6 else "Below Avg"
                csw_val = loc_ptd["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"]).mean() * 100
                st.markdown(
                    f'<div style="background:#f0f7ff;border-left:4px solid #2d7fc1;padding:12px 16px;'
                    f'border-radius:4px;margin:8px 0;">'
                    f'<b style="color:#1a1a2e;">Command Assessment — {loc_pitch_sel}:</b><br>'
                    f'<span style="color:#333;">Location spread: <b>{spread:.2f} ft ({grade})</b> | '
                    f'CSW%: <b>{csw_val:.1f}%</b> | '
                    f'Avg location: ({mean_s:.2f}S, {mean_h:.2f}H)<br>'
                    f'{"This pitch is located with precision — trust it in any count." if grade in ("Elite", "Plus") else "Work on tightening location consistency — wider spread means more mistakes over the plate."}'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )

    # ═══════════════════════════════════════════
    # TAB 7: AI REPORT
    # ═══════════════════════════════════════════
    with tab_ai:
        section_header("AI Scouting Report")
        st.caption("Comprehensive AI-generated analysis with actionable recommendations")

        tunnel_df_report = _compute_tunnel_score(pdf)
        pair_df_report = _compute_pitch_pair_results(pdf, data)
        report = _generate_ai_report(pdf, pitcher, stuff_df, tunnel_df_report, pair_df_report, data)
        st.markdown(report)

        # Downloadable report
        st.download_button(
            "Download Report as Text",
            report,
            file_name=f"pitch_design_report_{pitcher.replace(', ', '_')}.md",
            mime="text/markdown",
            key="pdl_download",
        )


# ──────────────────────────────────────────────
# HITTERS LAB HELPERS
# ──────────────────────────────────────────────

def _create_zone_grid_data(df, metric="swing_rate"):
    """Create 5x5 zone grid data for heatmaps."""
    h_edges = [-2, -0.83, -0.28, 0.28, 0.83, 2]
    v_edges = [0.5, 1.5, 2.17, 2.83, 3.5, 4.5]
    grid = np.full((5, 5), np.nan)
    annot = [['' for _ in range(5)] for _ in range(5)]
    for i in range(5):
        for j in range(5):
            zone_df = df[
                (df["PlateLocSide"].between(h_edges[i], h_edges[i + 1])) &
                (df["PlateLocHeight"].between(v_edges[j], v_edges[j + 1]))
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
    return grid, annot


def _compute_expected_outcomes(batted_df):
    """Compute expected outcomes based on EV/LA buckets."""
    if batted_df.empty:
        return {}
    outcomes = []
    for _, row in batted_df.iterrows():
        ev, la = row.get("ExitSpeed", 0), row.get("Angle", 0)
        if pd.isna(ev) or pd.isna(la):
            continue
        if ev >= 98 and 8 <= la <= 32:
            outcomes.append({"xOut": 0.25, "x1B": 0.10, "x2B": 0.20, "x3B": 0.05, "xHR": 0.40})
        elif ev >= 95 and 25 <= la <= 45:
            outcomes.append({"xOut": 0.40, "x1B": 0.05, "x2B": 0.15, "x3B": 0.05, "xHR": 0.35})
        elif 10 <= la <= 25 and ev >= 85:
            outcomes.append({"xOut": 0.30, "x1B": 0.45, "x2B": 0.20, "x3B": 0.03, "xHR": 0.02})
        elif la < 10:
            outcomes.append({"xOut": 0.75, "x1B": 0.23, "x2B": 0.02, "x3B": 0.00, "xHR": 0.00})
        elif la > 45:
            outcomes.append({"xOut": 0.95, "x1B": 0.03, "x2B": 0.01, "x3B": 0.00, "xHR": 0.01})
        elif ev < 70:
            outcomes.append({"xOut": 0.90, "x1B": 0.08, "x2B": 0.02, "x3B": 0.00, "xHR": 0.00})
        else:
            outcomes.append({"xOut": 0.70, "x1B": 0.20, "x2B": 0.08, "x3B": 0.01, "xHR": 0.01})
    if not outcomes:
        return {}
    odf = pd.DataFrame(outcomes)
    odf["xwOBA"] = 0.9 * odf["x1B"] + 1.25 * odf["x2B"] + 1.6 * odf["x3B"] + 2.0 * odf["xHR"]
    return odf.mean().to_dict()


def _generate_hitter_ai_report(bdf, batter_name, all_data, season_filter):
    """Generate a template-based AI scouting report for a hitter."""
    lines = []
    dn = display_name(batter_name)
    lines.append(f"# AI Hitting Report: {dn}")
    lines.append("")

    all_stats = compute_batter_stats(all_data, season_filter=season_filter)
    pr = all_stats[all_stats["Batter"] == batter_name]
    if pr.empty:
        lines.append("Insufficient data to generate report.")
        return "\n".join(lines)
    pr = pr.iloc[0]

    batted = bdf[bdf["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
    n_batted = len(batted)
    avg_ev = pr.get("AvgEV", np.nan)
    max_ev = pr.get("MaxEV", np.nan)
    barrel_pct = pr.get("BarrelPct", np.nan)
    hh_pct = pr.get("HardHitPct", np.nan)
    ss_pct = pr.get("SweetSpotPct", np.nan)
    k_pct = pr.get("KPct", np.nan)
    bb_pct = pr.get("BBPct", np.nan)
    whiff_pct = pr.get("WhiffPct", np.nan)
    chase_pct = pr.get("ChasePct", np.nan)
    gb_pct = pr.get("GBPct", np.nan)
    fb_pct = pr.get("FBPct", np.nan)
    ld_pct = pr.get("LDPct", np.nan)
    pull_pct = pr.get("PullPct", np.nan)
    oppo_pct = pr.get("OppoPct", np.nan)

    if not pd.isna(avg_ev) and avg_ev >= 89 and not pd.isna(barrel_pct) and barrel_pct >= 8:
        profile = "Power Hitter"
    elif not pd.isna(k_pct) and k_pct < 15 and not pd.isna(whiff_pct) and whiff_pct < 20:
        profile = "Contact-First Hitter"
    elif not pd.isna(bb_pct) and bb_pct >= 12 and not pd.isna(chase_pct) and chase_pct < 25:
        profile = "Disciplined Hitter"
    else:
        profile = "All-Around Hitter"

    side = bdf["BatterSide"].mode().iloc[0] if bdf["BatterSide"].notna().any() else "Unknown"
    bats = {"Right": "R", "Left": "L", "Switch": "S"}.get(side, side)

    lines.append(f"## Offensive Profile: {profile}")
    lines.append(f"- **Bats**: {bats}")
    lines.append(f"- **Pitches Seen**: {len(bdf)} | **Batted Balls**: {n_batted}")
    lines.append(f"- **PA**: {int(pr.get('PA', 0))}")
    lines.append("")

    lines.append("## Batted Ball Quality")
    ev_grade = "Elite" if not pd.isna(avg_ev) and avg_ev >= 91 else "Plus" if not pd.isna(avg_ev) and avg_ev >= 88 else "Average" if not pd.isna(avg_ev) and avg_ev >= 85 else "Below Average"
    lines.append(f"- **Avg Exit Velo**: {avg_ev:.1f} mph ({ev_grade})" if not pd.isna(avg_ev) else "- **Avg Exit Velo**: N/A")
    lines.append(f"- **Max Exit Velo**: {max_ev:.1f} mph" if not pd.isna(max_ev) else "- **Max Exit Velo**: N/A")
    lines.append(f"- **Barrel%**: {barrel_pct:.1f}%" if not pd.isna(barrel_pct) else "- **Barrel%**: N/A")
    lines.append(f"- **Hard-Hit%**: {hh_pct:.1f}%" if not pd.isna(hh_pct) else "- **Hard-Hit%**: N/A")
    lines.append(f"- **Sweet Spot%**: {ss_pct:.1f}%" if not pd.isna(ss_pct) else "- **Sweet Spot%**: N/A")
    lines.append("")

    lines.append("## Batted Ball Profile")
    if not any(pd.isna(x) for x in [gb_pct, ld_pct, fb_pct]):
        lines.append(f"- **GB%**: {gb_pct:.1f}% | **LD%**: {ld_pct:.1f}% | **FB%**: {fb_pct:.1f}%")
    else:
        lines.append("- Insufficient batted ball data")
    if not any(pd.isna(x) for x in [pull_pct, oppo_pct]):
        lines.append(f"- **Pull%**: {pull_pct:.1f}% | **Oppo%**: {oppo_pct:.1f}%")
    lines.append("")

    lines.append("## Plate Discipline")
    if not any(pd.isna(x) for x in [k_pct, bb_pct]):
        lines.append(f"- **K%**: {k_pct:.1f}% | **BB%**: {bb_pct:.1f}%")
    if not pd.isna(whiff_pct):
        lines.append(f"- **Whiff%**: {whiff_pct:.1f}%")
    if not pd.isna(chase_pct):
        lines.append(f"- **Chase%**: {chase_pct:.1f}%")
        if chase_pct > 35:
            lines.append("  - *High chase rate — susceptible to pitches out of the zone*")
        elif chase_pct < 22:
            lines.append("  - *Excellent discipline — rarely chases out of zone*")
    lines.append("")

    # Best / worst pitch types
    pt_evs = {}
    for pt in bdf["TaggedPitchType"].dropna().unique():
        pt_batted = bdf[(bdf["TaggedPitchType"] == pt) & (bdf["PitchCall"] == "InPlay")].dropna(subset=["ExitSpeed"])
        if len(pt_batted) >= 3:
            pt_evs[pt] = pt_batted["ExitSpeed"].mean()

    lines.append("## Strengths")
    strengths = []
    if not pd.isna(avg_ev) and avg_ev >= 88:
        strengths.append(f"Premium exit velocity ({avg_ev:.1f} mph avg)")
    if not pd.isna(barrel_pct) and barrel_pct >= 8:
        strengths.append(f"High barrel rate ({barrel_pct:.1f}%)")
    if not pd.isna(chase_pct) and chase_pct < 25:
        strengths.append(f"Elite plate discipline ({chase_pct:.1f}% chase)")
    if not pd.isna(bb_pct) and bb_pct >= 12:
        strengths.append(f"Strong walk rate ({bb_pct:.1f}%)")
    if not pd.isna(ld_pct) and ld_pct >= 25:
        strengths.append(f"Line drive machine ({ld_pct:.1f}% LD)")
    if not pd.isna(hh_pct) and hh_pct >= 40:
        strengths.append(f"Hard-hit rate ({hh_pct:.1f}%)")
    if pt_evs:
        best_pt = max(pt_evs, key=pt_evs.get)
        strengths.append(f"Best vs **{best_pt}** ({pt_evs[best_pt]:.1f} mph avg EV)")
    if not strengths:
        strengths.append("Developing hitter — building strengths across the board")
    for s in strengths:
        lines.append(f"- {s}")
    lines.append("")

    lines.append("## Areas for Improvement")
    weaknesses = []
    if not pd.isna(k_pct) and k_pct > 25:
        weaknesses.append(f"High strikeout rate ({k_pct:.1f}%)")
    if not pd.isna(chase_pct) and chase_pct > 32:
        weaknesses.append(f"Elevated chase rate ({chase_pct:.1f}%)")
    if not pd.isna(whiff_pct) and whiff_pct > 30:
        weaknesses.append(f"High whiff rate ({whiff_pct:.1f}%)")
    if not pd.isna(gb_pct) and gb_pct > 55:
        weaknesses.append(f"Ground ball heavy ({gb_pct:.1f}%) — needs to elevate")
    if not pd.isna(avg_ev) and avg_ev < 83:
        weaknesses.append(f"Below-average exit velocity ({avg_ev:.1f} mph)")
    if pt_evs:
        worst_pt = min(pt_evs, key=pt_evs.get)
        if pt_evs[worst_pt] < 85:
            weaknesses.append(f"Struggles vs **{worst_pt}** ({pt_evs[worst_pt]:.1f} mph avg EV)")
    if not weaknesses:
        weaknesses.append("No major weaknesses identified")
    for w in weaknesses:
        lines.append(f"- {w}")
    lines.append("")

    lines.append("## Scouting Report (Pitcher's Perspective)")
    game_plan = []
    if not pd.isna(chase_pct) and chase_pct > 30:
        game_plan.append("Expand the zone early — hitter chases frequently")
    if not pd.isna(whiff_pct) and whiff_pct > 28:
        game_plan.append("Use swing-and-miss pitches to get strikeouts")
    if pt_evs:
        worst_pt = min(pt_evs, key=pt_evs.get)
        game_plan.append(f"Attack with **{worst_pt}** — lowest damage pitch ({pt_evs[worst_pt]:.1f} mph avg EV)")
    if not pd.isna(pull_pct) and pull_pct > 50:
        game_plan.append(f"Heavy pull tendency ({pull_pct:.1f}%) — pitch away and shift")
    if not pd.isna(gb_pct) and gb_pct > 50:
        game_plan.append(f"Ground ball tendency ({gb_pct:.1f}%) — keep the ball down")
    if not game_plan:
        game_plan.append("Well-rounded hitter — mix pitches and locations")
    for g in game_plan:
        lines.append(f"- {g}")
    lines.append("")

    lines.append("## Development Recommendations")
    recs = []
    if not pd.isna(gb_pct) and gb_pct > 50:
        recs.append("Focus on launch angle — tee work emphasizing driving the ball in the air")
    if not pd.isna(chase_pct) and chase_pct > 30:
        recs.append("Pitch recognition drills — improve ability to lay off out-of-zone pitches")
    if not pd.isna(whiff_pct) and whiff_pct > 28:
        recs.append("Contact drills — focus on barrel accuracy and timing")
    if not pd.isna(avg_ev) and avg_ev < 85:
        recs.append("Bat speed training — increase exit velocity through strength and mechanics")
    if not pd.isna(oppo_pct) and oppo_pct < 15:
        recs.append("Opposite field approach — practice staying through the ball")
    if not recs:
        recs.append("Continue refining current approach — maintain strengths while looking for marginal gains")
    for r in recs:
        lines.append(f"- {r}")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# PAGE: HITTERS LAB
# ──────────────────────────────────────────────
def page_hitters_lab(data):
    st.markdown('<div class="section-header">Hitters Lab</div>', unsafe_allow_html=True)
    st.caption("Advanced hitting analytics: batted ball quality, plate discipline, zone coverage, approach optimization, and AI scouting")

    dav_hitting = filter_davidson(data, role="batter")
    if dav_hitting.empty:
        st.warning("No Davidson hitting data found.")
        return

    batters = sorted(dav_hitting["Batter"].unique())
    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        batter = st.selectbox("Select Hitter", batters, format_func=display_name, key="hl_batter")
    with col_sel2:
        seasons = sorted(dav_hitting["Season"].dropna().unique())
        season_filter = st.multiselect("Season", seasons, default=seasons, key="hl_season")

    bdf = dav_hitting[dav_hitting["Batter"] == batter].copy()
    if season_filter:
        bdf = bdf[bdf["Season"].isin(season_filter)]

    if len(bdf) < 20:
        st.warning("Not enough pitches (need 20+) to analyze.")
        return

    jersey = JERSEY.get(batter, "")
    pos = POSITION.get(batter, "")
    side = bdf["BatterSide"].mode().iloc[0] if bdf["BatterSide"].notna().any() else ""
    bats_str = {"Right": "R", "Left": "L", "Switch": "S"}.get(side, side)

    player_header(batter, jersey, pos,
                  f"{pos} | Bats: {bats_str} | Davidson Wildcats",
                  f"{len(bdf):,} pitches faced | Seasons: {', '.join(str(int(s)) for s in sorted(bdf['Season'].dropna().unique()))}")

    # Pre-compute common data
    batted = bdf[bdf["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
    in_zone_mask = (bdf["PlateLocSide"].abs() <= 0.83) & (bdf["PlateLocHeight"].between(1.5, 3.5))
    out_zone_mask = ~in_zone_mask & bdf["PlateLocSide"].notna() & bdf["PlateLocHeight"].notna()

    all_batter_stats = compute_batter_stats(data, season_filter=season_filter)
    player_row = all_batter_stats[all_batter_stats["Batter"] == batter]
    if player_row.empty:
        st.warning("Insufficient PA to compute stats.")
        return
    pr = player_row.iloc[0]

    tab_quality, tab_discipline, tab_coverage, tab_approach, tab_pitch_type, tab_spray, tab_swing, tab_ai = st.tabs([
        "Batted Ball Quality", "Plate Discipline", "Zone Coverage",
        "Approach Analysis", "Pitch Type Performance", "Spray Lab", "Swing Path", "AI Report"
    ])

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
                    (bp["ExitSpeed"] >= 98) & (bp["Angle"].between(8, 32)),
                    bp["ExitSpeed"] >= 95,
                    bp["ExitSpeed"].between(80, 95),
                ]
                bp["Quality"] = np.select(conditions, ["Barrel", "Hard-Hit", "Medium"], default="Weak")
                q_colors = {"Barrel": "#d22d49", "Hard-Hit": "#fe6100", "Medium": "#f7c631", "Weak": "#aaaaaa"}
                fig_ev = px.scatter(bp, x="Angle", y="ExitSpeed", color="Quality",
                                    color_discrete_map=q_colors,
                                    labels={"Angle": "Launch Angle", "ExitSpeed": "Exit Velocity (mph)"})
                fig_ev.add_shape(type="rect", x0=8, x1=32, y0=98, y1=batted["ExitSpeed"].max() + 5,
                                 fillcolor="rgba(210,45,73,0.08)", line=dict(color="rgba(210,45,73,0.3)", width=1, dash="dash"))
                fig_ev.add_annotation(x=20, y=batted["ExitSpeed"].max() + 3, text="Barrel Zone",
                                       font=dict(size=9, color="#d22d49"), showarrow=False)
                fig_ev.update_layout(**CHART_LAYOUT, height=400,
                                      legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font=dict(size=10)))
                st.plotly_chart(fig_ev, use_container_width=True)

            col_ev_dist, col_la_dist = st.columns(2)
            with col_ev_dist:
                section_header("Exit Velocity Distribution")
                all_batted = data[data["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
                fig_violin = go.Figure()
                fig_violin.add_trace(go.Violin(y=all_batted["ExitSpeed"], name="All Hitters",
                                                box_visible=True, meanline_visible=True,
                                                fillcolor="rgba(158,158,158,0.3)", line_color="#9e9e9e", opacity=0.6))
                fig_violin.add_trace(go.Violin(y=batted["ExitSpeed"], name=display_name(batter),
                                                box_visible=True, meanline_visible=True,
                                                fillcolor="rgba(210,45,73,0.4)", line_color="#d22d49", opacity=0.8))
                fig_violin.update_layout(**CHART_LAYOUT, height=320, showlegend=True,
                                          yaxis_title="Exit Velocity (mph)",
                                          legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font=dict(size=10)))
                st.plotly_chart(fig_violin, use_container_width=True)

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
                    ("x3B", "x3B%", "#ff7f0e"), ("xHR", "xHR%", "#d22d49"), ("xwOBA", "xwOBA", "#6a0dad"),
                ]):
                    with xo_cols[i]:
                        val = xo.get(k, 0)
                        fmt_val = f"{val*100:.1f}%" if k != "xwOBA" else f"{val:.3f}"
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
            grid_swing, annot_swing = _create_zone_grid_data(bdf, metric="swing_rate")
            h_labels = ["Far In", "Inside", "Middle", "Outside", "Far Out"]
            v_labels = ["Low+", "Low", "Mid", "High", "High+"]
            fig_grid = go.Figure(data=go.Heatmap(
                z=grid_swing, text=annot_swing, texttemplate="%{text}",
                x=h_labels, y=v_labels,
                colorscale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                zmin=0, zmax=100, showscale=True,
                colorbar=dict(title="Swing%", len=0.8),
            ))
            fig_grid.add_shape(type="rect", x0=0.5, x1=3.5, y0=0.5, y1=3.5,
                                line=dict(color="#333", width=3), fillcolor="rgba(0,0,0,0)")
            fig_grid.update_layout(**CHART_LAYOUT, height=380, xaxis=dict(side="bottom"))
            st.plotly_chart(fig_grid, use_container_width=True)

        col_ev_grid, col_chase = st.columns(2)
        with col_ev_grid:
            section_header("Avg EV by Zone")
            grid_ev, annot_ev = _create_zone_grid_data(bdf, metric="avg_ev")
            fig_ev_grid = go.Figure(data=go.Heatmap(
                z=grid_ev, text=annot_ev, texttemplate="%{text}",
                x=h_labels, y=v_labels,
                colorscale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                zmin=60, zmax=100, showscale=True,
                colorbar=dict(title="EV", len=0.8),
            ))
            fig_ev_grid.add_shape(type="rect", x0=0.5, x1=3.5, y0=0.5, y1=3.5,
                                   line=dict(color="#333", width=3), fillcolor="rgba(0,0,0,0)")
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
                                         xaxis=dict(range=[-2.5, 2.5]), yaxis=dict(range=[0, 5]),
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
            pt_oz = pt_df[~((pt_df["PlateLocSide"].abs() <= 0.83) & (pt_df["PlateLocHeight"].between(1.5, 3.5))) & pt_df["PlateLocSide"].notna()]
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
                                               xaxis=dict(range=[-2.5, 2.5], title="Horizontal"),
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
                    barrel_loc = batted_loc[(batted_loc["ExitSpeed"] >= 98) & (batted_loc["Angle"].between(8, 32))]
                    if not barrel_loc.empty:
                        fig_damage.add_trace(go.Scatter(
                            x=barrel_loc["PlateLocSide"], y=barrel_loc["PlateLocHeight"],
                            mode="markers", marker=dict(size=12, color="#d22d49", symbol="star",
                                                         line=dict(width=1, color="white")),
                            name="Barrels", hovertemplate="EV: %{customdata[0]:.1f}<extra></extra>",
                            customdata=barrel_loc[["ExitSpeed"]].values))
                    add_strike_zone(fig_damage)
                    fig_damage.update_layout(**CHART_LAYOUT, height=400,
                                              xaxis=dict(range=[-2.5, 2.5], title="Horizontal"),
                                              yaxis=dict(range=[0, 5], title="Vertical"),
                                              legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font=dict(size=10)))
                    st.plotly_chart(fig_damage, use_container_width=True)

            col_whiff_hz, _ = st.columns(2)
            with col_whiff_hz:
                section_header("Whiff Zone Map")
                grid_whiff, annot_whiff = _create_zone_grid_data(bdf, metric="whiff_rate")
                h_lbl = ["Far In", "Inside", "Middle", "Outside", "Far Out"]
                v_lbl = ["Low+", "Low", "Mid", "High", "High+"]
                fig_wz = go.Figure(data=go.Heatmap(
                    z=grid_whiff, text=annot_whiff, texttemplate="%{text}",
                    x=h_lbl, y=v_lbl,
                    colorscale=[[0, "#2ca02c"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                    zmin=0, zmax=60, showscale=True,
                    colorbar=dict(title="Whiff%", len=0.8)))
                fig_wz.add_shape(type="rect", x0=0.5, x1=3.5, y0=0.5, y1=3.5,
                                  line=dict(color="#333", width=3), fillcolor="rgba(0,0,0,0)")
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
            pt_br = pt_bt[(pt_bt["ExitSpeed"] >= 98) & (pt_bt["Angle"].between(8, 32))] if len(pt_bt) > 0 else pd.DataFrame()
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
                    bs = bdf["BatterSide"].mode().iloc[0] if bdf["BatterSide"].notna().any() else "Right"
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
                        brr = len(sub[(sub["ExitSpeed"] >= 98) & (sub["Angle"].between(8, 32))]) if sub["Angle"].notna().any() else 0
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
                    bs = bdf["BatterSide"].mode().iloc[0] if bdf["BatterSide"].notna().any() else "Right"
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
        st.caption("Reconstructed swing plane from contact quality, whiff locations, and approach angles — no bat sensor needed")

        swings = bdf[bdf["PitchCall"].isin(SWING_CALLS)].copy()
        whiffs = bdf[bdf["PitchCall"] == "StrikeSwinging"].copy()
        contacts = bdf[bdf["PitchCall"].isin(CONTACT_CALLS)].copy()
        inplay = bdf[(bdf["PitchCall"] == "InPlay")].copy()
        inplay_ev = inplay.dropna(subset=["ExitSpeed", "PlateLocSide", "PlateLocHeight"])

        if len(swings) < 10:
            st.info("Not enough swing data (need 10+ swings).")
        else:
            # ── Attack Angle Estimation ──
            section_header("Estimated Attack Angle")
            st.caption("Attack Angle ≈ Launch Angle − Vertical Approach Angle. Positive = upward bat path (lift). Negative = downward (chop).")

            inplay_aa = inplay.dropna(subset=["Angle", "VertApprAngle"]).copy()
            if len(inplay_aa) >= 5:
                inplay_aa["AttackAngle"] = inplay_aa["Angle"] - inplay_aa["VertApprAngle"]

                avg_aa = inplay_aa["AttackAngle"].mean()
                avg_la = inplay_aa["Angle"].mean()
                avg_vaa = inplay_aa["VertApprAngle"].mean()

                # Classify swing type
                if avg_aa > 15:
                    swing_type = "Steep Uppercut"
                    swing_color = "#d22d49"
                    swing_desc = "Extreme loft — high HR potential but vulnerable to high fastballs and off-speed below the zone"
                elif avg_aa > 8:
                    swing_type = "Lift-Oriented"
                    swing_color = "#fe6100"
                    swing_desc = "Modern swing path — good launch angle generation, solid barrel coverage of the zone"
                elif avg_aa > 2:
                    swing_type = "Slight Uppercut"
                    swing_color = "#f7c631"
                    swing_desc = "Balanced path with mild lift — matches average pitch plane well"
                elif avg_aa > -3:
                    swing_type = "Level"
                    swing_color = "#2ca02c"
                    swing_desc = "Flat bat path through the zone — contact-oriented, line drive approach"
                else:
                    swing_type = "Downward / Chopper"
                    swing_color = "#1f77b4"
                    swing_desc = "Downward swing plane — generates ground balls, limits hard fly ball contact"

                col_aa1, col_aa2, col_aa3 = st.columns(3)
                with col_aa1:
                    st.metric("Avg Attack Angle", f"{avg_aa:+.1f}°")
                with col_aa2:
                    st.metric("Avg Launch Angle", f"{avg_la:+.1f}°")
                with col_aa3:
                    st.metric("Avg Pitch VAA", f"{avg_vaa:.1f}°")

                st.markdown(
                    f'<div style="padding:12px 16px;background:white;border-radius:8px;border-left:5px solid {swing_color};'
                    f'border:1px solid #eee;margin:8px 0;">'
                    f'<span style="font-size:18px;font-weight:900;color:{swing_color} !important;">{swing_type}</span>'
                    f'<div style="font-size:13px;color:#333 !important;margin-top:4px;">{swing_desc}</div>'
                    f'</div>', unsafe_allow_html=True)

                # Attack angle by pitch type
                section_header("Attack Angle by Pitch Type")
                aa_pt_rows = []
                for pt in sorted(inplay_aa["TaggedPitchType"].dropna().unique()):
                    pt_df = inplay_aa[inplay_aa["TaggedPitchType"] == pt]
                    if len(pt_df) < 3:
                        continue
                    aa_pt_rows.append({
                        "Pitch Type": pt,
                        "n": len(pt_df),
                        "Avg Attack Angle": f"{pt_df['AttackAngle'].mean():+.1f}°",
                        "Avg LA": f"{pt_df['Angle'].mean():+.1f}°",
                        "Avg VAA": f"{pt_df['VertApprAngle'].mean():.1f}°",
                        "Avg EV": f"{pt_df['ExitSpeed'].mean():.1f}" if pt_df["ExitSpeed"].notna().any() else "-",
                    })
                if aa_pt_rows:
                    st.dataframe(pd.DataFrame(aa_pt_rows).set_index("Pitch Type"), use_container_width=True)

                # Attack angle distribution
                col_aa_dist, col_aa_height = st.columns(2)
                with col_aa_dist:
                    section_header("Attack Angle Distribution")
                    fig_aa = go.Figure()
                    fig_aa.add_trace(go.Histogram(
                        x=inplay_aa["AttackAngle"], nbinsx=25,
                        marker_color=swing_color, opacity=0.8, name="Attack Angle",
                    ))
                    fig_aa.add_vline(x=0, line_dash="dash", line_color="#888",
                                     annotation_text="Level swing", annotation_position="top")
                    fig_aa.add_vline(x=avg_aa, line_dash="solid", line_color="#1a1a2e",
                                     annotation_text=f"Avg: {avg_aa:+.1f}°", annotation_position="top right")
                    fig_aa.update_layout(**CHART_LAYOUT, height=300, xaxis_title="Attack Angle (°)",
                                          yaxis_title="Count", showlegend=False)
                    st.plotly_chart(fig_aa, use_container_width=True)

                with col_aa_height:
                    section_header("Attack Angle by Pitch Height")
                    st.caption("Does the bat path adjust to pitch location?")
                    fig_aa_h = go.Figure()
                    fig_aa_h.add_trace(go.Scatter(
                        x=inplay_aa["PlateLocHeight"], y=inplay_aa["AttackAngle"],
                        mode="markers",
                        marker=dict(size=6, color=inplay_aa["ExitSpeed"].fillna(80),
                                    colorscale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                                    cmin=60, cmax=105, showscale=True,
                                    colorbar=dict(title="EV", len=0.6),
                                    line=dict(width=0.3, color="white")),
                        hovertemplate="Height: %{x:.2f}ft<br>Attack: %{y:.1f}°<br>EV: %{marker.color:.1f}<extra></extra>",
                    ))
                    # Trend line
                    if len(inplay_aa) >= 10:
                        z = np.polyfit(inplay_aa["PlateLocHeight"].values, inplay_aa["AttackAngle"].values, 1)
                        x_line = np.linspace(inplay_aa["PlateLocHeight"].min(), inplay_aa["PlateLocHeight"].max(), 50)
                        fig_aa_h.add_trace(go.Scatter(x=x_line, y=np.polyval(z, x_line), mode="lines",
                                                       line=dict(color="#1a1a2e", width=2, dash="dash"),
                                                       name="Trend", showlegend=False))
                    fig_aa_h.add_hline(y=0, line_dash="dot", line_color="#ccc")
                    fig_aa_h.update_layout(**CHART_LAYOUT, height=300,
                                            xaxis_title="Pitch Height (ft)", yaxis_title="Attack Angle (°)",
                                            showlegend=False)
                    st.plotly_chart(fig_aa_h, use_container_width=True)

            else:
                st.info("Not enough InPlay pitches with launch angle + VAA data.")

            # ── Barrel Zone Map ──
            section_header("Barrel Path — EV Heatmap")
            st.caption("Where the barrel sweeps through the zone. High EV = barrel center. Low EV = handle/cap contact.")
            if len(inplay_ev) >= 10:
                col_barrel, col_whiff_path = st.columns(2)
                with col_barrel:
                    # EV heatmap using contour
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
                    # Overlay barrel contacts as stars
                    barrels = inplay_ev[(inplay_ev["ExitSpeed"] >= 98) & (inplay_ev["Angle"].between(8, 32))] if "Angle" in inplay_ev.columns else pd.DataFrame()
                    if len(barrels) > 0:
                        fig_barrel.add_trace(go.Scatter(
                            x=barrels["PlateLocSide"], y=barrels["PlateLocHeight"],
                            mode="markers", marker=dict(size=10, color="#d22d49", symbol="star",
                                                         line=dict(width=1, color="white")),
                            name="Barrels", showlegend=True,
                        ))
                    add_strike_zone(fig_barrel)
                    fig_barrel.update_layout(**CHART_LAYOUT, height=420,
                                              xaxis=dict(range=[-2.5, 2.5], title="Horizontal"),
                                              yaxis=dict(range=[0, 5], title="Vertical"),
                                              title="Contact Quality (EV) by Location",
                                              legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"))
                    st.plotly_chart(fig_barrel, use_container_width=True)

                with col_whiff_path:
                    # Whiff density — where the bat ISN'T
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
                        # Overlay contact locations faintly
                        con_loc = contacts.dropna(subset=["PlateLocSide", "PlateLocHeight"])
                        if len(con_loc) > 0:
                            fig_whiff.add_trace(go.Scatter(
                                x=con_loc["PlateLocSide"], y=con_loc["PlateLocHeight"],
                                mode="markers", marker=dict(size=3, color="#2ca02c", opacity=0.3),
                                name="Contact", showlegend=True,
                            ))
                        add_strike_zone(fig_whiff)
                        fig_whiff.update_layout(**CHART_LAYOUT, height=420,
                                                  xaxis=dict(range=[-2.5, 2.5], title="Horizontal"),
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
                                           xaxis=dict(range=[-2.5, 2.5], title="Horizontal"),
                                           yaxis=dict(range=[0, 5], title="Vertical"),
                                           title="All Pitches: Swing vs Take",
                                           legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"))
                    st.plotly_chart(fig_dec, use_container_width=True)

                with col_dec2:
                    # Swing decision contour — probability of swinging at each location
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
                                            xaxis=dict(range=[-2.5, 2.5], title="Horizontal"),
                                            yaxis=dict(range=[0, 5], title="Vertical"),
                                            title="Swing Probability by Location")
                    st.plotly_chart(fig_prob, use_container_width=True)

            # ── Swing Path Summary Card ──
            section_header("Swing Path Profile")
            swing_n = len(swings)
            whiff_n = len(whiffs)
            contact_n = len(contacts)
            inplay_n = len(inplay_ev)

            # Compute zone-level stats for the profile
            high_swings = swings[swings["PlateLocHeight"] > 2.83] if "PlateLocHeight" in swings.columns else pd.DataFrame()
            low_swings = swings[swings["PlateLocHeight"] < 2.17] if "PlateLocHeight" in swings.columns else pd.DataFrame()
            high_whiff_rate = len(high_swings[high_swings["PitchCall"] == "StrikeSwinging"]) / max(len(high_swings), 1) * 100
            low_whiff_rate = len(low_swings[low_swings["PitchCall"] == "StrikeSwinging"]) / max(len(low_swings), 1) * 100

            high_ev = inplay_ev[inplay_ev["PlateLocHeight"] > 2.83]["ExitSpeed"].mean() if len(inplay_ev[inplay_ev["PlateLocHeight"] > 2.83]) >= 3 else np.nan
            low_ev = inplay_ev[inplay_ev["PlateLocHeight"] < 2.17]["ExitSpeed"].mean() if len(inplay_ev[inplay_ev["PlateLocHeight"] < 2.17]) >= 3 else np.nan

            insights = []
            if not pd.isna(high_ev) and not pd.isna(low_ev):
                if high_ev > low_ev + 3:
                    insights.append(f"Barrel is **higher in the zone** — {high_ev:.1f} mph (high) vs {low_ev:.1f} mph (low). Swing plane sits up.")
                elif low_ev > high_ev + 3:
                    insights.append(f"Barrel is **lower in the zone** — {low_ev:.1f} mph (low) vs {high_ev:.1f} mph (high). Swing plane sits down.")
                else:
                    insights.append(f"Even EV top-to-bottom ({high_ev:.1f} high vs {low_ev:.1f} low) — good vertical barrel coverage.")

            if high_whiff_rate > low_whiff_rate + 10:
                insights.append(f"More vulnerable **up** ({high_whiff_rate:.0f}% whiff high vs {low_whiff_rate:.0f}% low) — bat path may sit below the high fastball.")
            elif low_whiff_rate > high_whiff_rate + 10:
                insights.append(f"More vulnerable **down** ({low_whiff_rate:.0f}% whiff low vs {high_whiff_rate:.0f}% high) — bat path may sweep over breaking balls.")

            # Inside vs outside
            inside_ev = inplay_ev[inplay_ev["PlateLocSide"] < -0.28]["ExitSpeed"].mean() if batter_side == "Right" else inplay_ev[inplay_ev["PlateLocSide"] > 0.28]["ExitSpeed"].mean()
            outside_ev = inplay_ev[inplay_ev["PlateLocSide"] > 0.28]["ExitSpeed"].mean() if batter_side == "Right" else inplay_ev[inplay_ev["PlateLocSide"] < -0.28]["ExitSpeed"].mean()
            if not pd.isna(inside_ev) and not pd.isna(outside_ev):
                if inside_ev > outside_ev + 3:
                    insights.append(f"Stronger **inside** ({inside_ev:.1f} mph) than outside ({outside_ev:.1f} mph) — barrel reaches inside pitch well.")
                elif outside_ev > inside_ev + 3:
                    insights.append(f"Stronger **outside** ({outside_ev:.1f} mph) than inside ({inside_ev:.1f} mph) — extends barrel well to the outer half.")

            if insights:
                for insight in insights:
                    st.markdown(f"- {insight}")
            else:
                st.info("Not enough zone-split data to generate swing path insights.")

    # ─── Tab 8: AI Scouting Report ─────────────────────
    with tab_ai:
        section_header("AI Hitting Report")
        report = _generate_hitter_ai_report(bdf, batter, data, season_filter)
        st.markdown(report)
        st.download_button(
            "Download Report as Text", report,
            file_name=f"hitting_report_{batter.replace(', ', '_')}.md",
            mime="text/markdown", key="hl_download")


# ──────────────────────────────────────────────
# PAGE: MATCHUP OPTIMIZER
# ──────────────────────────────────────────────
def page_matchup_optimizer(data):
    st.markdown('<div class="section-header">Matchup Optimizer</div>', unsafe_allow_html=True)
    st.caption("Rank Davidson hitters by expected performance against a specific opposing pitcher's arsenal")

    # Select opponent pitcher
    opp_pitching = data[~data["Pitcher"].isin(ROSTER_2026) & data["PitcherTeam"].notna()].copy()
    if opp_pitching.empty:
        st.warning("No opponent pitching data found.")
        return

    opp_pitchers = sorted(opp_pitching["Pitcher"].unique())
    col1, col2 = st.columns([2, 1])
    with col1:
        opp_pitcher = st.selectbox("Select Opposing Pitcher", opp_pitchers,
                                    format_func=display_name, key="mo_pitcher")
    with col2:
        seasons = sorted(data["Season"].dropna().unique())
        season_filter = st.multiselect("Season", seasons, default=seasons, key="mo_season")

    opdf = opp_pitching[opp_pitching["Pitcher"] == opp_pitcher]
    if season_filter:
        opdf = opdf[opdf["Season"].isin(season_filter)]
    if len(opdf) < 10:
        st.warning("Not enough data for this pitcher (need 10+ pitches).")
        return

    throws = opdf["PitcherThrows"].mode().iloc[0] if opdf["PitcherThrows"].notna().any() else "?"
    team = opdf["PitcherTeam"].mode().iloc[0] if opdf["PitcherTeam"].notna().any() else "?"

    # Pitcher arsenal summary
    section_header(f"{display_name(opp_pitcher)} — {team} ({throws}HP) — {len(opdf)} pitches")

    arsenal = []
    for pt in sorted(opdf["TaggedPitchType"].dropna().unique()):
        pt_df = opdf[opdf["TaggedPitchType"] == pt]
        pct = len(pt_df) / len(opdf) * 100
        avg_velo = pt_df["RelSpeed"].mean() if pt_df["RelSpeed"].notna().any() else np.nan
        arsenal.append({"Pitch": pt, "Usage": f"{pct:.0f}%", "Avg Velo": f"{avg_velo:.1f}" if not pd.isna(avg_velo) else "-",
                        "usage_num": pct})
    arsenal_df = pd.DataFrame(arsenal).sort_values("usage_num", ascending=False).drop(columns=["usage_num"])
    st.dataframe(arsenal_df.set_index("Pitch"), use_container_width=True)

    # Score each Davidson hitter vs this pitcher's pitch type mix
    section_header("Hitter Rankings vs This Pitcher")
    dav_hitting = filter_davidson(data, role="batter")
    if season_filter:
        dav_hitting = dav_hitting[dav_hitting["Season"].isin(season_filter)]

    pitch_mix = opdf["TaggedPitchType"].value_counts(normalize=True)

    hitter_scores = []
    for batter in dav_hitting["Batter"].unique():
        bdf = dav_hitting[dav_hitting["Batter"] == batter]
        if len(bdf) < 20:
            continue
        side = bdf["BatterSide"].mode().iloc[0] if bdf["BatterSide"].notna().any() else "?"
        bats = {"Right": "R", "Left": "L", "Switch": "S"}.get(side, side)

        # Weighted score across pitch types
        weighted_ev = 0
        weighted_whiff = 0
        weighted_chase = 0
        total_weight = 0
        pt_details = {}

        for pt, pct in pitch_mix.items():
            pt_bdf = bdf[bdf["TaggedPitchType"] == pt]
            if len(pt_bdf) < 3:
                continue
            sw = pt_bdf[pt_bdf["PitchCall"].isin(SWING_CALLS)]
            wh = pt_bdf[pt_bdf["PitchCall"] == "StrikeSwinging"]
            bt = pt_bdf[pt_bdf["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            oz = pt_bdf[~((pt_bdf["PlateLocSide"].abs() <= 0.83) & (pt_bdf["PlateLocHeight"].between(1.5, 3.5))) & pt_bdf["PlateLocSide"].notna()]
            ch = oz[oz["PitchCall"].isin(SWING_CALLS)]

            avg_ev = bt["ExitSpeed"].mean() if len(bt) > 0 else 75
            whiff_rate = len(wh) / max(len(sw), 1) * 100 if len(sw) > 0 else 50
            chase_rate = len(ch) / max(len(oz), 1) * 100 if len(oz) > 0 else 30

            weighted_ev += avg_ev * pct
            weighted_whiff += whiff_rate * pct
            weighted_chase += chase_rate * pct
            total_weight += pct
            pt_details[pt] = {"EV": f"{avg_ev:.1f}", "Whiff": f"{whiff_rate:.0f}%"}

        if total_weight < 0.3:
            continue

        weighted_ev /= total_weight
        weighted_whiff /= total_weight
        weighted_chase /= total_weight

        # Composite score: higher EV good, lower whiff good, lower chase good
        composite = (weighted_ev - 75) * 2 - weighted_whiff * 0.5 - weighted_chase * 0.3

        # Platoon advantage
        platoon = ""
        if throws == "Right" and bats == "L":
            platoon = "Platoon+"
        elif throws == "Left" and bats == "R":
            platoon = "Platoon+"
        elif throws == "Right" and bats == "R":
            platoon = "Same"
        elif throws == "Left" and bats == "L":
            platoon = "Same"

        hitter_scores.append({
            "Hitter": batter,
            "Bats": bats,
            "Platoon": platoon,
            "Pitches": len(bdf),
            "xEV": round(weighted_ev, 1),
            "xWhiff%": round(weighted_whiff, 1),
            "xChase%": round(weighted_chase, 1),
            "Score": round(composite, 1),
            "_score": composite,
        })

    if hitter_scores:
        score_df = pd.DataFrame(hitter_scores).sort_values("_score", ascending=False)

        # Color-code ranks
        st.markdown("*Higher Score = better matchup. Score combines expected EV, whiff rate, and chase rate weighted by pitcher's pitch mix.*")

        display_scores = score_df.drop(columns=["_score"]).copy()
        display_scores.insert(0, "Rank", range(1, len(display_scores) + 1))
        display_scores["Hitter"] = display_scores["Hitter"].apply(display_name)
        st.dataframe(display_scores.set_index("Rank"), use_container_width=True, height=min(len(display_scores) * 40 + 50, 600))

        # Top 9 lineup recommendation
        section_header("Recommended Lineup (Top 9)")
        top9 = score_df.head(9)
        lineup_cols = st.columns(3)
        for i, (_, row) in enumerate(top9.iterrows()):
            with lineup_cols[i % 3]:
                clr = "#2ca02c" if row["Score"] > 10 else "#f7c631" if row["Score"] > 0 else "#d22d49"
                st.markdown(
                    f'<div style="padding:10px;background:white;border-radius:8px;border:1px solid #eee;margin:4px 0;'
                    f'border-left:4px solid {clr};">'
                    f'<div style="font-size:20px;font-weight:800;color:#1a1a2e !important;">#{i+1}</div>'
                    f'<div style="font-size:14px;font-weight:700;color:#1a1a2e !important;">{display_name(row["Hitter"])}</div>'
                    f'<div style="font-size:11px;color:#666 !important;">Bats: {row["Bats"]} | {row["Platoon"]} | '
                    f'xEV: {row["xEV"]} | Score: {row["Score"]}</div></div>', unsafe_allow_html=True)

        # Detailed matchup breakdown for selected hitter
        section_header("Detailed Matchup Breakdown")
        selected = st.selectbox("Select Hitter for Detail", score_df["Hitter"].tolist(),
                                 format_func=display_name, key="mo_detail")
        sel_bdf = dav_hitting[dav_hitting["Batter"] == selected]
        detail_rows = []
        for pt in sorted(opdf["TaggedPitchType"].dropna().unique()):
            pt_usage = len(opdf[opdf["TaggedPitchType"] == pt]) / len(opdf) * 100
            pt_bdf = sel_bdf[sel_bdf["TaggedPitchType"] == pt]
            if len(pt_bdf) < 3:
                detail_rows.append({"Pitch": pt, "Opp Usage": f"{pt_usage:.0f}%", "Seen": len(pt_bdf),
                                     "Swing%": "-", "Whiff%": "-", "Avg EV": "-", "Barrel%": "-"})
                continue
            sw = pt_bdf[pt_bdf["PitchCall"].isin(SWING_CALLS)]
            wh = pt_bdf[pt_bdf["PitchCall"] == "StrikeSwinging"]
            bt = pt_bdf[pt_bdf["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            br = bt[(bt["ExitSpeed"] >= 98) & (bt["Angle"].between(8, 32))] if len(bt) > 0 else pd.DataFrame()
            detail_rows.append({
                "Pitch": pt,
                "Opp Usage": f"{pt_usage:.0f}%",
                "Seen": len(pt_bdf),
                "Swing%": f"{len(sw)/len(pt_bdf)*100:.1f}%",
                "Whiff%": f"{len(wh)/max(len(sw),1)*100:.1f}%" if len(sw) > 0 else "-",
                "Avg EV": f"{bt['ExitSpeed'].mean():.1f}" if len(bt) > 0 else "-",
                "Barrel%": f"{len(br)/max(len(bt),1)*100:.1f}%" if len(bt) > 0 else "-",
            })
        if detail_rows:
            st.dataframe(pd.DataFrame(detail_rows).set_index("Pitch"), use_container_width=True)
    else:
        st.info("Not enough Davidson hitting data against these pitch types.")


# ──────────────────────────────────────────────
# PAGE: GAME PLANNING
# ──────────────────────────────────────────────
def page_game_planning(data):
    st.markdown('<div class="section-header">Game Planning</div>', unsafe_allow_html=True)
    st.caption("Pitch sequencing engine, count leverage analysis, and effective velocity — actionable intel for game day")

    dav_pitching = filter_davidson(data, role="pitcher")
    if dav_pitching.empty:
        st.warning("No Davidson pitching data found.")
        return

    pitchers = sorted(dav_pitching["Pitcher"].unique())
    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        pitcher = st.selectbox("Select Pitcher", pitchers, format_func=display_name, key="gp_pitcher")
    with col_sel2:
        seasons = sorted(dav_pitching["Season"].dropna().unique())
        season_filter = st.multiselect("Season", seasons, default=seasons, key="gp_season")

    pdf = dav_pitching[dav_pitching["Pitcher"] == pitcher].copy()
    if season_filter:
        pdf = pdf[pdf["Season"].isin(season_filter)]
    if len(pdf) < 30:
        st.warning("Not enough pitches (need 30+).")
        return
    pdf = filter_minor_pitches(pdf)

    jersey = JERSEY.get(pitcher, "")
    pos = POSITION.get(pitcher, "")
    throws = pdf["PitcherThrows"].mode().iloc[0] if pdf["PitcherThrows"].notna().any() else ""
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
        tunnel_df = _compute_tunnel_score(pdf)
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
                st.caption("Physics-based: pitches that look identical at the commit point (~167ms before plate) but diverge at the plate")
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
                    if len(pair) < 3:
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
                    if len(pair) < 5:
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
                    # Composite: 50% whiff normalized (0-50% → 0-100) + 30% tunnel score + 20% inverse EV
                    whiff_norm = min(whiff_pct / 40.0, 1.0) * 100
                    tunnel_norm = tunnel_score if not pd.isna(tunnel_score) else 50
                    ev_val = bt["NextEV"].mean() if len(bt) > 0 else np.nan
                    ev_norm = max(0, min(100, (105 - ev_val) / 25 * 100)) if not pd.isna(ev_val) else 50
                    combo_score = whiff_norm * 0.50 + tunnel_norm * 0.30 + ev_norm * 0.20
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
                with st.expander("Full Sequence + Tunnel Table"):
                    disp_seq = seq_df.drop(columns=["_whiff", "_combo"]).copy()
                    for c in ["Swing%", "Whiff%", "CSW%"]:
                        disp_seq[c] = disp_seq[c].map(lambda x: f"{x:.1f}%")
                    disp_seq["Avg EV"] = disp_seq["Avg EV"].map(lambda x: f"{x:.1f}" if not pd.isna(x) else "-")
                    disp_seq["Tunnel Score"] = disp_seq["Tunnel Score"].map(lambda x: f"{x:.0f}" if not pd.isna(x) else "-")
                    disp_seq["Combo Score"] = disp_seq["Combo Score"].map(lambda x: f"{x:.0f}")
                    disp_seq = disp_seq.sort_values("Combo Score", ascending=False)
                    st.dataframe(disp_seq.set_index("Sequence"), use_container_width=True)
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
        selected_count = st.selectbox("Select Count", [f"{b}-{s}" for b in range(4) for s in range(3)], key="gp_count")
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
                "Zone%": f"{len(sit_df[(sit_df['PlateLocSide'].abs() <= 0.83) & (sit_df['PlateLocHeight'].between(1.5, 3.5))])/max(len(sit_df[sit_df['PlateLocSide'].notna()]),1)*100:.1f}%",
                "Whiff%": f"{len(wh)/max(len(sw),1)*100:.1f}%",
                "CSW%": f"{len(csw)/len(sit_df)*100:.1f}%",
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
                                        xaxis=dict(range=[-2.5, 2.5], title="Horizontal"),
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
                                        xaxis=dict(range=[-2.5, 2.5], title="Horizontal"),
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
        # All batters in the database
        batters = sorted(data[data["PitchCall"] == "InPlay"]["Batter"].dropna().unique())
        if not batters:
            st.warning("No batted ball data found.")
            return
        col1, col2 = st.columns([2, 1])
        with col1:
            batter = st.selectbox("Select Batter", batters, format_func=display_name, key="dp_batter")
        with col2:
            seasons = sorted(data["Season"].dropna().unique())
            season_filter = st.multiselect("Season", seasons, default=seasons, key="dp_season")
        bdf = data[(data["Batter"] == batter)].copy()
        if season_filter:
            bdf = bdf[bdf["Season"].isin(season_filter)]
        label = display_name(batter)
    else:
        teams = sorted(data["BatterTeam"].dropna().unique())
        col1, col2 = st.columns([2, 1])
        with col1:
            team = st.selectbox("Select Team", teams, key="dp_team")
        with col2:
            seasons = sorted(data["Season"].dropna().unique())
            season_filter = st.multiselect("Season", seasons, default=seasons, key="dp_season_t")
        bdf = data[data["BatterTeam"] == team].copy()
        if season_filter:
            bdf = bdf[bdf["Season"].isin(season_filter)]
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
                        "Hit%": f"{z['count']/max(len(filt),1)*100:.1f}%",
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
            gb_pull_pct = len(gb_pull) / max(len(spray_all[spray_all["TaggedHitType"] == "GroundBall"]), 1) * 100 if "TaggedHitType" in spray_all.columns else 0

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
# MAIN
# ──────────────────────────────────────────────
def main():
    st.sidebar.markdown(
        '<div class="sidebar-brand" style="text-align:center;padding:10px 0 5px 0;">'
        '<div class="sidebar-brand-title" style="font-size:22px;font-weight:800;font-family:Inter,sans-serif;">'
        'Davidson Baseball</div>'
        '<div class="sidebar-brand-sub" style="font-size:11px;letter-spacing:1px;text-transform:uppercase;'
        'font-family:Inter,sans-serif;">Trackman Analytics</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    page = st.sidebar.radio("Navigation", [
        "Hitter Card",
        "Pitcher Card",
        "Catcher Analytics",
        "Team Overview",
        "Player Development",
        "Pitch Design Lab",
        "Hitters Lab",
        "Matchup Optimizer",
        "Game Planning",
        "Defensive Positioning",
        "Game Log",
        "Opponent Scouting",
    ], label_visibility="collapsed")

    data = load_all_data()
    if data.empty:
        st.error("No data loaded.")
        return

    st.sidebar.markdown("---")
    st.sidebar.markdown(f'<div style="font-size:12px;color:#888 !important;padding:0 10px;">'
                        f'<b style="color:#cc0000 !important;">{len(data):,}</b> pitches<br>'
                        f'<b style="color:#cc0000 !important;">{data["Season"].nunique()}</b> seasons '
                        f'({int(data["Season"].min())}-{int(data["Season"].max())})<br>'
                        f'<b style="color:#cc0000 !important;">{len(ROSTER_2026)}</b> rostered players'
                        f'</div>', unsafe_allow_html=True)

    if page == "Hitter Card":
        page_hitter_card(data)
    elif page == "Pitcher Card":
        page_pitcher_card(data)
    elif page == "Catcher Analytics":
        page_catcher(data)
    elif page == "Team Overview":
        page_team(data)
    elif page == "Player Development":
        page_development(data)
    elif page == "Pitch Design Lab":
        page_pitch_design_lab(data)
    elif page == "Hitters Lab":
        page_hitters_lab(data)
    elif page == "Matchup Optimizer":
        page_matchup_optimizer(data)
    elif page == "Game Planning":
        page_game_planning(data)
    elif page == "Defensive Positioning":
        page_defensive_positioning(data)
    elif page == "Game Log":
        page_game_log(data)
    elif page == "Opponent Scouting":
        page_scouting(data)


if __name__ == "__main__":
    main()
