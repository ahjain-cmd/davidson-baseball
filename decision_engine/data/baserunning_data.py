from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import _APP_DIR


def _first_existing(*paths: str) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _read_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        # Fall back without encoding hint.
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()


def _pct_to_float(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        if s.endswith("%"):
            try:
                return float(s[:-1])
            except Exception:
                return np.nan
        if s in {"-", ""}:
            return np.nan
    return v


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    out = df.copy()
    for c in out.columns:
        if out[c].dtype != object:
            continue
        sample = out[c].dropna().head(25)
        if sample.empty:
            continue
        # Percent columns
        if sample.astype(str).str.strip().str.endswith("%").mean() > 0.5:
            out[c] = out[c].apply(_pct_to_float)
            out[c] = pd.to_numeric(out[c], errors="coerce")
            continue
        # Numeric-looking strings
        conv = pd.to_numeric(sample, errors="coerce")
        if conv.notna().mean() > 0.8:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _norm_player_id(v: Any) -> Optional[int]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        # pandas may parse large ids as floats; preserve by casting via int(str)
        if isinstance(v, float):
            return int(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        s = str(v).strip()
        if not s:
            return None
        return int(float(s)) if "." in s else int(s)
    except Exception:
        return None


def _data_path(rel: str) -> str:
    return os.path.join(_APP_DIR, "data", rel)


@lru_cache(maxsize=1)
def load_speed_scores() -> pd.DataFrame:
    path = _first_existing(_data_path("speed_scores.csv"), os.path.join(_APP_DIR, "Speed Score-2.csv"))
    df = _coerce_numeric(_read_csv(path or ""))
    if df.empty:
        return df
    if "playerId" in df.columns:
        df["playerId"] = df["playerId"].apply(_norm_player_id)
    return df


@lru_cache(maxsize=1)
def load_stolen_bases_hitters() -> pd.DataFrame:
    path = _first_existing(_data_path("stolen_bases_hitters.csv"), os.path.join(_APP_DIR, "Stolen Bases hitters.csv"))
    df = _coerce_numeric(_read_csv(path or ""))
    if df.empty:
        return df
    if "playerId" in df.columns:
        df["playerId"] = df["playerId"].apply(_norm_player_id)
    return df


@lru_cache(maxsize=1)
def load_count_sb_squeeze() -> pd.DataFrame:
    path = _first_existing(_data_path("count_sb_squeeze.csv"), os.path.join(_APP_DIR, "Count SB SQ.csv"))
    df = _coerce_numeric(_read_csv(path or ""))
    if df.empty:
        return df
    if "playerId" in df.columns:
        df["playerId"] = df["playerId"].apply(_norm_player_id)
    return df


@lru_cache(maxsize=1)
def load_stolen_bases_catchers() -> pd.DataFrame:
    path = _first_existing(_data_path("stolen_bases_catchers.csv"), os.path.join(_APP_DIR, "Stolen Bases-3.csv"))
    df = _coerce_numeric(_read_csv(path or ""))
    if df.empty:
        return df
    if "playerId" in df.columns:
        df["playerId"] = df["playerId"].apply(_norm_player_id)
    return df


@lru_cache(maxsize=1)
def load_stolen_bases_pitchers() -> pd.DataFrame:
    path = _first_existing(_data_path("stolen_bases_pitchers.csv"), os.path.join(_APP_DIR, "Stolen Bases pitching.csv"))
    df = _coerce_numeric(_read_csv(path or ""))
    if df.empty:
        return df
    if "playerId" in df.columns:
        df["playerId"] = df["playerId"].apply(_norm_player_id)
    return df


@lru_cache(maxsize=1)
def load_pickoffs() -> pd.DataFrame:
    path = _first_existing(_data_path("pickoffs.csv"), os.path.join(_APP_DIR, "Pickoffs.csv"))
    df = _coerce_numeric(_read_csv(path or ""))
    if df.empty:
        return df
    if "playerId" in df.columns:
        df["playerId"] = df["playerId"].apply(_norm_player_id)
    return df


@lru_cache(maxsize=1)
def load_running_bases() -> pd.DataFrame:
    path = _first_existing(_data_path("running_bases.csv"), os.path.join(_APP_DIR, "Running Bases.csv"))
    df = _coerce_numeric(_read_csv(path or ""))
    return df


@lru_cache(maxsize=1)
def _speed_by_player_id() -> Dict[int, Dict[str, Any]]:
    df = load_speed_scores()
    if df.empty or "playerId" not in df.columns:
        return {}
    out = {}
    for _, row in df.dropna(subset=["playerId"]).iterrows():
        pid = _norm_player_id(row.get("playerId"))
        if pid is None:
            continue
        out[pid] = row.to_dict()
    return out


@lru_cache(maxsize=1)
def _sb_hit_by_player_id() -> Dict[int, Dict[str, Any]]:
    df = load_stolen_bases_hitters()
    if df.empty or "playerId" not in df.columns:
        return {}
    out = {}
    for _, row in df.dropna(subset=["playerId"]).iterrows():
        pid = _norm_player_id(row.get("playerId"))
        if pid is None:
            continue
        out[pid] = row.to_dict()
    return out


@lru_cache(maxsize=1)
def _sq_by_player_id() -> Dict[int, Dict[str, Any]]:
    df = load_count_sb_squeeze()
    if df.empty or "playerId" not in df.columns:
        return {}
    out = {}
    for _, row in df.dropna(subset=["playerId"]).iterrows():
        pid = _norm_player_id(row.get("playerId"))
        if pid is None:
            continue
        out[pid] = row.to_dict()
    return out


@lru_cache(maxsize=1)
def _catcher_by_name() -> Dict[str, Dict[str, Any]]:
    df = load_stolen_bases_catchers()
    if df.empty or "playerFullName" not in df.columns:
        return {}
    out = {}
    for _, row in df.dropna(subset=["playerFullName"]).iterrows():
        nm = str(row["playerFullName"]).strip().lower()
        if not nm:
            continue
        out[nm] = row.to_dict()
    return out


@lru_cache(maxsize=1)
def _pitcher_sb_by_name() -> Dict[str, Dict[str, Any]]:
    df = load_stolen_bases_pitchers()
    if df.empty or "playerFullName" not in df.columns:
        return {}
    out = {}
    for _, row in df.dropna(subset=["playerFullName"]).iterrows():
        nm = str(row["playerFullName"]).strip().lower()
        if not nm:
            continue
        out[nm] = row.to_dict()
    return out


@lru_cache(maxsize=1)
def _pickoffs_by_name() -> Dict[str, Dict[str, Any]]:
    df = load_pickoffs()
    if df.empty or "playerFullName" not in df.columns:
        return {}
    out = {}
    for _, row in df.dropna(subset=["playerFullName"]).iterrows():
        nm = str(row["playerFullName"]).strip().lower()
        if not nm:
            continue
        out[nm] = row.to_dict()
    return out


def get_speed_row(player_id: Optional[int]) -> Optional[Dict[str, Any]]:
    if player_id is None:
        return None
    return _speed_by_player_id().get(int(player_id))


def get_stolen_bases_row(player_id: Optional[int]) -> Optional[Dict[str, Any]]:
    if player_id is None:
        return None
    return _sb_hit_by_player_id().get(int(player_id))


def get_squeeze_row(player_id: Optional[int]) -> Optional[Dict[str, Any]]:
    if player_id is None:
        return None
    return _sq_by_player_id().get(int(player_id))


def get_catcher_row_by_name(full_name: str) -> Optional[Dict[str, Any]]:
    if not full_name:
        return None
    return _catcher_by_name().get(str(full_name).strip().lower())


def get_pitcher_sb_row_by_name(full_name: str) -> Optional[Dict[str, Any]]:
    if not full_name:
        return None
    return _pitcher_sb_by_name().get(str(full_name).strip().lower())


def get_pickoffs_row_by_name(full_name: str) -> Optional[Dict[str, Any]]:
    if not full_name:
        return None
    return _pickoffs_by_name().get(str(full_name).strip().lower())


def speed_score_and_sb_from_sources(
    *,
    player_id: Optional[int],
    counting_row: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], Optional[int], Optional[float]]:
    """Return (SpeedScore, SB count, SB%) with local-file primary + pack fallback."""
    speed_row = get_speed_row(player_id)
    if speed_row:
        ss = speed_row.get("SpeedScore")
        sb = speed_row.get("SB")
        sbp = speed_row.get("SB%")
        try:
            ss = float(ss) if ss is not None and not (isinstance(ss, float) and np.isnan(ss)) else None
        except Exception:
            ss = None
        try:
            sb = int(sb) if sb is not None and not (isinstance(sb, float) and np.isnan(sb)) else None
        except Exception:
            sb = None
        try:
            sbp = float(sbp) if sbp is not None and not (isinstance(sbp, float) and np.isnan(sbp)) else None
        except Exception:
            sbp = None
        return ss, sb, sbp

    # Fallback to counting stats when SpeedScore export doesn't contain this player.
    if counting_row:
        sb = counting_row.get("SB")
        try:
            sb_i = int(sb) if sb is not None and not (isinstance(sb, float) and np.isnan(sb)) else None
        except Exception:
            sb_i = None
        return None, sb_i, None

    return None, None, None


def pitcher_sb_pct_from_trackman(df: pd.DataFrame, pitcher_name: str, min_attempts: int = 3) -> Optional[float]:
    """Compute SB% allowed from Trackman PlayResult (StolenBase/CaughtStealing)."""
    if df is None or df.empty:
        return None
    if "Pitcher" not in df.columns or "PlayResult" not in df.columns:
        return None
    sub = df[(df["Pitcher"] == pitcher_name) & (df["PlayResult"].isin(["StolenBase", "CaughtStealing"]))].copy()
    if sub.empty:
        return None
    sb = int((sub["PlayResult"] == "StolenBase").sum())
    cs = int((sub["PlayResult"] == "CaughtStealing").sum())
    att = sb + cs
    if att < int(min_attempts):
        return None
    return float(sb / att * 100.0)


def pitcher_sb_summary_from_trackman(df: pd.DataFrame, pitcher_name: str) -> Optional[Dict[str, Any]]:
    """Return {"sb":..,"cs":..,"att":..,"sb_pct":..} from Trackman PlayResult."""
    if df is None or df.empty:
        return None
    if "Pitcher" not in df.columns or "PlayResult" not in df.columns:
        return None
    sub = df[(df["Pitcher"] == pitcher_name) & (df["PlayResult"].isin(["StolenBase", "CaughtStealing"]))].copy()
    if sub.empty:
        return {"sb": 0, "cs": 0, "att": 0, "sb_pct": None}
    sb = int((sub["PlayResult"] == "StolenBase").sum())
    cs = int((sub["PlayResult"] == "CaughtStealing").sum())
    att = sb + cs
    sb_pct = float(sb / att * 100.0) if att > 0 else None
    return {"sb": sb, "cs": cs, "att": att, "sb_pct": sb_pct}
