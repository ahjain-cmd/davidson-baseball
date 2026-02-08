"""Build a combined Bryant 2026 opponent pack from multi-year, multi-school data.

For returning Bryant players: fetches Bryant 2024 + 2025 packs.
For transfers: fetches their previous school's 2024 + 2025 packs.
Filters each pack to only the matching roster player, concatenates, deduplicates
(keeping most recent season when both exist), and caches as a synthetic pack.
"""
from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np

from config import (
    CACHE_DIR,
    BRYANT_TEAM_NAME, BRYANT_ROSTER_2026, BRYANT_TRANSFERS,
    BRYANT_COMBINED_TEAM_ID, display_name,
)
from data.truemedia_api import fetch_all_teams, build_tm_dict_for_team
from decision_engine.data.opponent_pack import (
    save_opponent_pack, load_opponent_pack, _meta_path,
    _HITTING_TABLES, _PITCHING_TABLES, _CATCHING_TABLES, _DEFENSE_TABLES,
)


_NAME_COL_CANDIDATES = ["playerFullName", "fullName"]

_SUFFIX_RE = None

def _norm_name(n: str) -> str:
    """Normalize a player name for fuzzy matching: lowercase, strip suffixes, collapse spaces."""
    global _SUFFIX_RE
    if _SUFFIX_RE is None:
        import re
        _SUFFIX_RE = re.compile(r'\s+(jr\.?|sr\.?|ii|iii|iv|v)\s*$', re.IGNORECASE)
    s = " ".join(str(n).lower().strip().split())
    s = _SUFFIX_RE.sub("", s)
    return s


def _first_last_key(n: str) -> str:
    """Extract 'first last' key from a full name (ignores middle names/suffixes)."""
    parts = _norm_name(n).split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[-1]}"
    return _norm_name(n)


# Common nicknames where the first initial differs from the legal name.
_NICKNAME_ALIASES = {
    "billy": "william", "bill": "william", "will": "william", "willy": "william",
    "bob": "robert", "bobby": "robert", "rob": "robert",
    "dick": "richard", "rick": "richard", "richie": "richard",
    "ted": "theodore", "teddy": "theodore",
    "jack": "john", "johnny": "john",
    "chuck": "charles", "charlie": "charles",
    "jim": "james", "jimmy": "james", "jamie": "james",
    "hank": "henry",
    "peggy": "margaret",
}
# Build reverse map too (william -> billy, etc.)
_LEGAL_TO_NICK = {}
for _nick, _legal in _NICKNAME_ALIASES.items():
    _LEGAL_TO_NICK.setdefault(_legal, set()).add(_nick)


def _last_initial_key(n: str) -> str:
    """Extract 'X lastname' key (first initial + last name) for nickname-tolerant matching.

    Handles common nickname mismatches like Mike/Michael, Cam/Cameron, Tommy/Thomas.
    """
    parts = _norm_name(n).split()
    if len(parts) >= 2:
        return f"{parts[0][0]} {parts[-1]}"
    return ""


def _nickname_keys(n: str) -> Set[str]:
    """Return set of all 'initial lastname' keys for a name, including nickname variants.

    For 'Billy Mulholland', returns {'b mulholland', 'w mulholland'} because
    Billy is a nickname for William.
    """
    parts = _norm_name(n).split()
    if len(parts) < 2:
        return set()
    first, last = parts[0], parts[-1]
    keys = {f"{first[0]} {last}"}
    # If first name is a known nickname, add legal name's initial
    legal = _NICKNAME_ALIASES.get(first)
    if legal:
        keys.add(f"{legal[0]} {last}")
    # If first name is a known legal name, add all nickname initials
    for nick in _LEGAL_TO_NICK.get(first, []):
        keys.add(f"{nick[0]} {last}")
    return keys


def _roster_display_names() -> Set[str]:
    """Return set of 'First Last' names for the Bryant 2026 roster."""
    return {display_name(n, escape_html=False) for n in BRYANT_ROSTER_2026}


def _find_team_id(teams_df: pd.DataFrame, search_name: str) -> Optional[Tuple[str, str]]:
    """Search a teams DataFrame for a team matching *search_name*.

    Uses progressively looser matching:
    1. Exact match on fullName (case-insensitive)
    2. Exact match on location column (case-insensitive)
    3. Contains match on fullName — but prefer shortest match (most specific)

    Returns (team_id, team_full_name) or None.
    """
    if teams_df is None or teams_df.empty:
        return None

    name_col = None
    for c in ["fullName", "newestTeamName", "teamName", "name"]:
        if c in teams_df.columns:
            name_col = c
            break
    if name_col is None:
        return None

    id_col = None
    for c in ["teamId", "id", "teamID"]:
        if c in teams_df.columns:
            id_col = c
            break
    if id_col is None:
        return None

    search_lower = search_name.strip().lower()

    # 1. Exact match on fullName
    exact = teams_df[teams_df[name_col].astype(str).str.strip().str.lower() == search_lower]
    if not exact.empty:
        return str(exact.iloc[0][id_col]), str(exact.iloc[0][name_col])

    # 2. Exact match on location column (if present)
    if "location" in teams_df.columns:
        loc_exact = teams_df[teams_df["location"].astype(str).str.strip().str.lower() == search_lower]
        if not loc_exact.empty:
            return str(loc_exact.iloc[0][id_col]), str(loc_exact.iloc[0][name_col])

    # 3. Contains match — prefer shortest fullName (most specific match)
    mask = teams_df[name_col].astype(str).str.contains(search_name, case=False, na=False)
    matches = teams_df[mask]
    if matches.empty:
        return None

    # Sort by name length so "Bryant University" beats "Bryant and Stratton College"
    matches = matches.copy()
    matches["_name_len"] = matches[name_col].astype(str).str.len()
    matches = matches.sort_values("_name_len")

    return str(matches.iloc[0][id_col]), str(matches.iloc[0][name_col])


def _build_name_matcher(player_names_display: Set[str]):
    """Build lookup dicts for robust name matching.

    Returns (exact_set, norm_set, first_last_set, initial_last_set) where each
    set contains normalized versions of the target names for progressively
    looser matching.  The last level matches on first initial + last name to
    handle nickname mismatches (Mike/Michael, Cam/Cameron, Tommy/Thomas, etc.).
    """
    exact = {n.strip() for n in player_names_display}
    norm = {_norm_name(n) for n in player_names_display}
    fl = {_first_last_key(n) for n in player_names_display}
    il: Set[str] = set()
    for n in player_names_display:
        il.update(_nickname_keys(n))
    il.discard("")
    return exact, norm, fl, il


def _name_matches(api_name: str, exact: Set[str], norm: Set[str],
                  fl: Set[str], il: Set[str]) -> bool:
    """Check if an API-returned name matches any target name.

    Match priority: exact → normalized → first+last → initial+last (with nicknames).
    The initial+last level matches "M fiatarone" to handle Mike/Michael mismatches.
    """
    s = str(api_name).strip()
    if s in exact:
        return True
    if _norm_name(s) in norm:
        return True
    if _first_last_key(s) in fl:
        return True
    if il:
        # Check all nickname variants of the API name against the target set
        for key in _nickname_keys(s):
            if key in il:
                return True
    return False


def _filter_pack_to_players(
    pack: Dict[str, Any],
    player_names_display: Set[str],
) -> Dict[str, Any]:
    """Filter all DataFrames in a pack to only rows matching the given player names.

    player_names_display should be in "First Last" format (matching playerFullName).
    Uses progressively looser matching: exact → normalized → first+last name only.
    """
    exact, norm, fl, il = _build_name_matcher(player_names_display)

    filtered: Dict[str, Any] = {}
    for group_name in ("hitting", "pitching", "catching", "defense"):
        group = pack.get(group_name, {})
        if not isinstance(group, dict):
            filtered[group_name] = {}
            continue
        filtered_group: Dict[str, Any] = {}
        for table_name, df in group.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                filtered_group[table_name] = df
                continue
            # Try to filter by player name
            matched = False
            for col in _NAME_COL_CANDIDATES:
                if col in df.columns:
                    mask = df[col].apply(lambda x: _name_matches(x, exact, norm, fl, il))
                    filtered_group[table_name] = df[mask].copy()
                    matched = True
                    break
            if not matched:
                # Table doesn't have player names (team-level data) — include as-is
                filtered_group[table_name] = df
        filtered[group_name] = filtered_group
    return filtered


def _merge_packs(packs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple filtered packs by concatenating their DataFrames.

    When a player appears in multiple packs (e.g., 2024 + 2025), keeps
    the row from the last pack in the list (most recent season) for rate stats,
    while summing counting stats.
    """
    # Initialize structure
    all_tables = {
        "hitting": _HITTING_TABLES,
        "pitching": _PITCHING_TABLES,
        "catching": _CATCHING_TABLES,
        "defense": _DEFENSE_TABLES,
    }
    merged: Dict[str, Any] = {}

    for group_name, table_names in all_tables.items():
        merged[group_name] = {}
        for table_name in table_names:
            frames = []
            for p in packs:
                df = p.get(group_name, {}).get(table_name)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    frames.append(df)
            if not frames:
                merged[group_name][table_name] = pd.DataFrame()
                continue

            combined = pd.concat(frames, ignore_index=True)

            # Deduplicate: keep last occurrence (most recent season) per player
            name_col = None
            for c in _NAME_COL_CANDIDATES:
                if c in combined.columns:
                    name_col = c
                    break
            if name_col:
                combined = combined.drop_duplicates(subset=[name_col], keep="last")

            merged[group_name][table_name] = combined

    return merged


def build_bryant_combined_pack(
    *,
    refresh: bool = False,
    seasons: List[int] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """Build a combined Bryant 2026 pack from 2024+2025 multi-school data.

    Parameters
    ----------
    refresh : bool
        Force re-fetch from API even if cached.
    seasons : list[int]
        Seasons to fetch. Default: [2024, 2025].
    progress_callback : callable, optional
        Called with (message: str) for UI progress updates.

    Returns
    -------
    dict
        Standard opponent pack structure with combined data.
    """
    if seasons is None:
        seasons = [2024, 2025]

    def _log(msg):
        if progress_callback:
            progress_callback(msg)

    # Check cache first
    if not refresh:
        cached = load_opponent_pack(BRYANT_COMBINED_TEAM_ID, season_year=2026)
        if cached is not None:
            _log("Loaded from cache")
            return cached

    roster_display = _roster_display_names()

    # Group players by source school.
    # ALL players are searched under Bryant University (they may have data there
    # from prior seasons even if they transferred in).  Transfer players are
    # ADDITIONALLY searched at their previous school.
    school_players: Dict[str, Set[str]] = {}  # school_search_name -> set of display names

    # Every roster player should be searched under Bryant
    all_display = set()
    for roster_name in BRYANT_ROSTER_2026:
        all_display.add(display_name(roster_name, escape_html=False))
    school_players[BRYANT_TEAM_NAME] = all_display

    # Additionally search transfer players at their previous schools
    for roster_name, prev_school in BRYANT_TRANSFERS.items():
        disp = display_name(roster_name, escape_html=False)
        school_players.setdefault(prev_school, set()).add(disp)

    # For each season, fetch teams list
    all_packs: List[Dict[str, Any]] = []  # ordered: earlier seasons first, so "last" = most recent

    for season in sorted(seasons):
        _log(f"Fetching team list for {season}...")
        teams_df = fetch_all_teams(season_year=season)
        if teams_df is None or teams_df.empty:
            _log(f"No teams found for {season}, skipping.")
            continue

        for school_search, player_set in school_players.items():
            result = _find_team_id(teams_df, school_search)
            if result is None:
                _log(f"  {school_search} not found in {season}")
                continue

            team_id, team_full_name = result
            _log(f"  {school_search} -> {team_full_name} ({team_id}) [{season}]")

            try:
                pack = build_tm_dict_for_team(team_id, team_full_name, season_year=season)
            except Exception as e:
                _log(f"  Failed to fetch {school_search} {season}: {e}")
                continue

            # Log available players from this team before filtering
            _unfiltered_h = pack.get("hitting", {}).get("rate")
            if isinstance(_unfiltered_h, pd.DataFrame) and not _unfiltered_h.empty:
                _ncol = "playerFullName" if "playerFullName" in _unfiltered_h.columns else "fullName"
                if _ncol in _unfiltered_h.columns:
                    api_names = _unfiltered_h[_ncol].dropna().astype(str).str.strip().tolist()
                    _log(f"    API has {len(api_names)} players: {api_names[:10]}{'...' if len(api_names) > 10 else ''}")
                    # Show which roster names we're looking for
                    _log(f"    Looking for: {sorted(player_set)[:10]}{'...' if len(player_set) > 10 else ''}")

            # Filter to only our roster players from this school
            filtered = _filter_pack_to_players(pack, player_set)

            # Check if we actually got any player data
            h_rate = filtered.get("hitting", {}).get("rate")
            p_trad = filtered.get("pitching", {}).get("traditional")
            n_hitters = len(h_rate) if isinstance(h_rate, pd.DataFrame) and not h_rate.empty else 0
            n_pitchers = len(p_trad) if isinstance(p_trad, pd.DataFrame) and not p_trad.empty else 0
            _log(f"    -> {n_hitters} hitters, {n_pitchers} pitchers matched")

            if n_hitters > 0 or n_pitchers > 0:
                all_packs.append(filtered)

    if not all_packs:
        _log("No data found for any Bryant player. Returning empty pack.")
        return {
            "hitting": {k: pd.DataFrame() for k in _HITTING_TABLES},
            "pitching": {k: pd.DataFrame() for k in _PITCHING_TABLES},
            "catching": {k: pd.DataFrame() for k in _CATCHING_TABLES},
            "defense": {k: pd.DataFrame() for k in _DEFENSE_TABLES},
        }

    # Merge all filtered packs
    _log("Merging data from all sources...")
    combined = _merge_packs(all_packs)

    # Cache it
    _log("Saving combined pack to cache...")
    save_opponent_pack(
        combined,
        team_id=BRYANT_COMBINED_TEAM_ID,
        team_name="Bryant (2024-25 Combined)",
        season_year=2026,
    )

    _log("Done!")
    return combined


def load_bryant_combined_pack() -> Optional[Dict[str, Any]]:
    """Load the cached Bryant combined pack, or None if not built yet."""
    return load_opponent_pack(BRYANT_COMBINED_TEAM_ID, season_year=2026)
