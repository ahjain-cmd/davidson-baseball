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
    elif page == "Game Log":
        page_game_log(data)
    elif page == "Opponent Scouting":
        page_scouting(data)


if __name__ == "__main__":
    main()
