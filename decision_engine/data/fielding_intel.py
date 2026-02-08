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
        if sample.astype(str).str.strip().str.endswith("%").mean() > 0.5:
            out[c] = out[c].apply(_pct_to_float)
            out[c] = pd.to_numeric(out[c], errors="coerce")
            continue
        conv = pd.to_numeric(sample, errors="coerce")
        if conv.notna().mean() > 0.8:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _data_path(rel: str) -> str:
    return os.path.join(_APP_DIR, "data", rel)


@lru_cache(maxsize=1)
def load_all_fielding_counting() -> pd.DataFrame:
    path = _first_existing(_data_path("all_fielding_counting.csv"), os.path.join(_APP_DIR, "All Fielding Counting.csv"))
    return _coerce_numeric(_read_csv(path or ""))


@lru_cache(maxsize=1)
def load_fielding_participation() -> pd.DataFrame:
    path = _first_existing(_data_path("fielding_participation.csv"), os.path.join(_APP_DIR, "Fielding Participation.csv"))
    return _coerce_numeric(_read_csv(path or ""))


@lru_cache(maxsize=1)
def load_outfield_throws() -> pd.DataFrame:
    path = _first_existing(_data_path("outfield_throws.csv"), os.path.join(_APP_DIR, "Outfield Throws.csv"))
    return _coerce_numeric(_read_csv(path or ""))


@lru_cache(maxsize=1)
def load_outfield_counting() -> pd.DataFrame:
    path = _first_existing(_data_path("outfield_counting.csv"), os.path.join(_APP_DIR, "Outfield Counting.csv"))
    return _coerce_numeric(_read_csv(path or ""))


@lru_cache(maxsize=1)
def _position_avg_range() -> Dict[str, float]:
    """Average chances per 9 innings by position, from the D1 fielding export."""
    df = load_all_fielding_counting()
    if df.empty:
        return {}
    req = {"pos", "Inn", "Chances"}
    if not req.issubset(df.columns):
        return {}
    sub = df.dropna(subset=["pos", "Inn", "Chances"]).copy()
    if sub.empty:
        return {}
    inn = pd.to_numeric(sub["Inn"], errors="coerce")
    chances = pd.to_numeric(sub["Chances"], errors="coerce")
    valid = inn.notna() & chances.notna() & (inn > 0)
    sub = sub[valid].copy()
    if sub.empty:
        return {}
    sub["_range_proxy"] = chances[valid] / (inn[valid] / 9.0).clip(lower=1.0)
    return sub.groupby("pos")["_range_proxy"].mean().to_dict()


def _get_fielding_benchmarks():
    """Lazy-load D1 fielding benchmarks from historical calibration."""
    try:
        from analytics.historical_calibration import load_historical_calibration, fallback_fielding_benchmarks
        cal = load_historical_calibration()
        return cal.fielding_benchmarks if cal else fallback_fielding_benchmarks()
    except Exception:
        return {"fld_pct_median": 0.9727, "fld_pct_p75": 0.980, "fld_pct_p90": 0.990,
                "error_rate_median": 0.0273, "n_bip": 0}


def _fielder_quality(row: Dict[str, Any], position_avg_range: Dict[str, float]) -> int:
    """Rate a fielder 1-5 based on FLD%, error rate, and a simple range proxy.

    Thresholds are calibrated from D1 fielding data (P90 for elite, P75 for good).
    """
    bench = _get_fielding_benchmarks()
    elite_fld = bench.get("fld_pct_p90", 0.990)
    good_fld = bench.get("fld_pct_p75", 0.980)

    fld = row.get("FLD%")
    try:
        fld_pct = float(fld) if fld is not None and pd.notna(fld) else np.nan
    except Exception:
        fld_pct = np.nan
    chances = row.get("Chances")
    e = row.get("E")
    inn = row.get("Inn")
    try:
        chances = float(chances) if chances is not None and pd.notna(chances) else 0.0
    except Exception:
        chances = 0.0
    try:
        e = float(e) if e is not None and pd.notna(e) else 0.0
    except Exception:
        e = 0.0
    try:
        inn = float(inn) if inn is not None and pd.notna(inn) else 0.0
    except Exception:
        inn = 0.0

    error_rate = e / max(chances, 1.0)
    range_proxy = chances / max(inn / 9.0, 1.0)  # chances per 9 innings

    score = 0
    if pd.notna(fld_pct):
        if fld_pct >= elite_fld:
            score += 2
        elif fld_pct >= good_fld:
            score += 1
    if error_rate <= bench.get("error_rate_median", 0.02):
        score += 1
    pos = str(row.get("pos", "")).strip()
    base = position_avg_range.get(pos)
    if base is not None and pd.notna(base) and range_proxy > float(base):
        score += 1
    return int(min(5, max(1, score)))


def _of_arm_rating(of_ast: float, inn: float) -> str:
    if inn < 50:
        return "Unknown"
    rate = of_ast / (inn / 100.0) if inn > 0 else 0.0
    if rate >= 3.0:
        return "Elite"
    if rate >= 1.5:
        return "Strong"
    if rate >= 0.5:
        return "Average"
    return "Weak"


def team_defense_profile(team_name: str) -> Dict[str, pd.DataFrame]:
    """Return team-level defensive intel tables for Module C."""
    team_name = str(team_name).strip()
    if not team_name:
        return {}

    field = load_all_fielding_counting()
    part = load_fielding_participation()
    of_thr = load_outfield_throws()
    of_cnt = load_outfield_counting()

    # Filter by newestTeamName when present; fall back to teamFullName/name columns.
    def _team_filter(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        for c in ["newestTeamName", "teamFullName", "teamName", "Team", "team"]:
            if c in df.columns:
                return df[df[c].astype(str).str.strip() == team_name].copy()
        return pd.DataFrame()

    field_t = _team_filter(field)
    part_t = _team_filter(part)
    of_thr_t = _team_filter(of_thr)
    of_cnt_t = _team_filter(of_cnt)

    pos_avg = _position_avg_range()

    # Primary fielders by position (highest innings at that pos).
    starters_rows = []
    if not field_t.empty and {"pos", "Inn"}.issubset(field_t.columns):
        for pos, grp in field_t.dropna(subset=["pos", "Inn"]).groupby("pos"):
            g = grp.copy()
            g["Inn"] = pd.to_numeric(g["Inn"], errors="coerce")
            g = g.dropna(subset=["Inn"])
            if g.empty:
                continue
            row = g.sort_values("Inn", ascending=False).iloc[0].to_dict()
            row["quality"] = _fielder_quality(row, pos_avg)
            # Basic attackability flag — threshold from D1 fielding median.
            fld_pct = row.get("FLD%")
            try:
                fld_pct = float(fld_pct) if fld_pct is not None and pd.notna(fld_pct) else np.nan
            except Exception:
                fld_pct = np.nan
            fld_bench = _get_fielding_benchmarks()
            attackable_threshold = fld_bench.get("fld_pct_median", 0.9727)
            if pd.notna(fld_pct) and fld_pct < attackable_threshold:
                row["note"] = "attackable (low FLD%)"
            else:
                row["note"] = ""
            starters_rows.append(row)
    starters = pd.DataFrame(starters_rows)

    # OF arm ratings: join OF throws with OF innings when possible.
    of_arms = pd.DataFrame()
    if not of_thr_t.empty:
        of_arms = of_thr_t.copy()
        if not of_cnt_t.empty and "playerId" in of_arms.columns and "playerId" in of_cnt_t.columns:
            of_arms = of_arms.merge(of_cnt_t[["playerId", "InnOF"]], on="playerId", how="left")
        inn = pd.to_numeric(of_arms.get("InnOF"), errors="coerce") if "InnOF" in of_arms.columns else pd.Series([np.nan] * len(of_arms))
        of_ast = pd.to_numeric(of_arms.get("OFAst"), errors="coerce") if "OFAst" in of_arms.columns else pd.Series([np.nan] * len(of_arms))
        ratings = []
        for i in range(len(of_arms)):
            inn_i = float(inn.iloc[i]) if pd.notna(inn.iloc[i]) else 0.0
            ast_i = float(of_ast.iloc[i]) if pd.notna(of_ast.iloc[i]) else 0.0
            ratings.append(_of_arm_rating(ast_i, inn_i))
        of_arms["ArmRating"] = ratings

    return {
        "fielding_counting": field_t,
        "fielding_participation": part_t,
        "outfield_throws": of_thr_t,
        "outfield_counting": of_cnt_t,
        "starters": starters,
        "of_arms": of_arms,
    }

