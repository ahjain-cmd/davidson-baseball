from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import pandas as pd

from config import CACHE_DIR
from data.truemedia_api import build_tm_dict_for_team

from decision_engine.data.baserunning_data import (
    load_count_sb_squeeze,
    load_running_bases,
    load_speed_scores,
    load_stolen_bases_hitters,
)
from decision_engine.data.fielding_intel import team_defense_profile


_HITTING_TABLES = [
    "rate",
    "counting",
    "exit",
    "expected_rate",
    "expected_hit_rates",
    "hit_types",
    "hit_locations",
    "pitch_rates",
    "pitch_types",
    "pitch_locations",
    "pitch_counts",
    "pitch_type_counts",
    "pitch_calls",
    "speed",
    "stolen_bases",
    "count_sb_squeeze",
    "running_bases_team",
    "home_runs",
    "run_expectancy",
    "swing_pct",
    "swing_stats",
]

_PITCHING_TABLES = [
    "traditional",
    "rate",
    "movement",
    "pitch_types",
    "pitch_rates",
    "exit",
    "expected_rate",
    "hit_types",
    "hit_locations",
    "counting",
    "pitch_counts",
    "pitch_locations",
    "baserunning",
    "stolen_bases",
    "home_runs",
    "expected_hit_rates",
    "pitch_calls",
    "pitch_type_counts",
    "expected_counting",
    "pitching_counting",
    "bids",
]

_CATCHING_TABLES = [
    "defense",
    "framing",
    "opposing",
    "pitch_rates",
    "pitch_types_rates",
    "pitch_types",
    "throws",
    "sba2_throws",
    "pickoffs",
    "pb_wp",
    "pitch_counts",
]

_DEFENSE_TABLES = [
    "fielding_counting",
    "fielding_participation",
    "outfield_throws",
    "outfield_counting",
    "starters",
    "of_arms",
]


def _pack_dir(team_id: str, season_year: int) -> str:
    safe_team = str(team_id).strip().replace("/", "_")
    return os.path.join(CACHE_DIR, "opponent_packs", str(int(season_year)), safe_team)


def _meta_path(team_id: str, season_year: int) -> str:
    return os.path.join(_pack_dir(team_id, season_year), "meta.json")


def _table_path(team_id: str, season_year: int, group: str, table: str) -> str:
    return os.path.join(_pack_dir(team_id, season_year), group, f"{table}.parquet")


def _is_stale(meta: Dict[str, Any], ttl_hours: float) -> bool:
    ts = meta.get("created_at_unix")
    if ts is None:
        return True
    try:
        age_h = (time.time() - float(ts)) / 3600.0
    except Exception:
        return True
    return age_h > float(ttl_hours)


def save_opponent_pack(pack: Dict[str, Any], team_id: str, team_name: str, season_year: int) -> None:
    root = _pack_dir(team_id, season_year)
    os.makedirs(root, exist_ok=True)

    meta = {
        "team_id": team_id,
        "team_name": team_name,
        "season_year": int(season_year),
        "created_at_unix": time.time(),
    }
    with open(_meta_path(team_id, season_year), "w", encoding="utf-8") as f:
        json.dump(meta, f)

    for group_name in ("hitting", "pitching", "catching", "defense"):
        group = pack.get(group_name, {}) if isinstance(pack, dict) else {}
        if not isinstance(group, dict):
            continue
        for table_name, df in group.items():
            if not isinstance(df, pd.DataFrame):
                continue
            if df.empty:
                continue
            out_path = _table_path(team_id, season_year, group_name, table_name)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            try:
                df.to_parquet(out_path, index=False)
            except Exception:
                # Cache is best-effort.
                pass


def load_opponent_pack(team_id: str, season_year: int) -> Optional[Dict[str, Any]]:
    meta_fp = _meta_path(team_id, season_year)
    if not os.path.exists(meta_fp):
        return None
    try:
        with open(meta_fp, "r", encoding="utf-8") as f:
            _ = json.load(f)  # meta currently unused by loader beyond existence
    except Exception:
        return None

    # Initialize full structure (downstream code expects keys to exist, even if empty).
    out: Dict[str, Any] = {
        "hitting": {k: pd.DataFrame() for k in _HITTING_TABLES},
        "pitching": {k: pd.DataFrame() for k in _PITCHING_TABLES},
        "catching": {k: pd.DataFrame() for k in _CATCHING_TABLES},
        "defense": {k: pd.DataFrame() for k in _DEFENSE_TABLES},
    }
    for group_name in ("hitting", "pitching", "catching", "defense"):
        group_dir = os.path.join(_pack_dir(team_id, season_year), group_name)
        if not os.path.isdir(group_dir):
            continue
        for fn in os.listdir(group_dir):
            if not fn.endswith(".parquet"):
                continue
            table_name = fn[: -len(".parquet")]
            fp = os.path.join(group_dir, fn)
            try:
                out[group_name][table_name] = pd.read_parquet(fp)
            except Exception:
                out[group_name][table_name] = pd.DataFrame()
    return out


def load_or_build_opponent_pack(
    *,
    team_id: str,
    team_name: str,
    season_year: int,
    refresh: bool = False,
    ttl_hours: float = 24.0,
) -> Dict[str, Any]:
    """Load an opponent pack from disk if present; otherwise build via TrueMedia API and cache."""
    meta_fp = _meta_path(team_id, season_year)
    if not refresh and os.path.exists(meta_fp):
        try:
            with open(meta_fp, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if not _is_stale(meta, ttl_hours=ttl_hours):
                cached = load_opponent_pack(team_id, season_year)
                if cached is not None:
                    return cached
        except Exception:
            pass

    pack = build_tm_dict_for_team(team_id, team_name, season_year=season_year)

    # Enrich pack with local D1 exports (baserunning + defense intel) so it works offline.
    try:
        spd = load_speed_scores()
        if not spd.empty and "newestTeamName" in spd.columns:
            pack.setdefault("hitting", {})["speed"] = spd[spd["newestTeamName"].astype(str).str.strip() == team_name].copy()
    except Exception:
        pass

    try:
        sbh = load_stolen_bases_hitters()
        if not sbh.empty and "newestTeamName" in sbh.columns:
            pack.setdefault("hitting", {})["stolen_bases"] = sbh[sbh["newestTeamName"].astype(str).str.strip() == team_name].copy()
    except Exception:
        pass

    try:
        sq = load_count_sb_squeeze()
        if not sq.empty and "newestTeamName" in sq.columns:
            pack.setdefault("hitting", {})["count_sb_squeeze"] = sq[sq["newestTeamName"].astype(str).str.strip() == team_name].copy()
    except Exception:
        pass

    try:
        rb = load_running_bases()
        if not rb.empty:
            # Running Bases is team-level; store as 1-row DF for this team when available.
            for c in ["newestTeamName", "teamFullName", "teamName", "team"]:
                if c in rb.columns:
                    pack.setdefault("hitting", {})["running_bases_team"] = rb[rb[c].astype(str).str.strip() == team_name].copy()
                    break
    except Exception:
        pass

    try:
        pack["defense"] = team_defense_profile(team_name)
    except Exception:
        pack["defense"] = {k: pd.DataFrame() for k in _DEFENSE_TABLES}

    save_opponent_pack(pack, team_id=team_id, team_name=team_name, season_year=season_year)
    return pack
