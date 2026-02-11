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


def _enrich_bats_from_pitches(rate_df: pd.DataFrame, pitches: pd.DataFrame) -> None:
    """Infer batsHand from BatterSide in pitch-level data for hitters missing it.

    Modifies rate_df in place — adds 'batsHand' column if not present and fills
    missing values from the most common BatterSide for each hitter.
    """
    if "Batter" not in pitches.columns or "BatterSide" not in pitches.columns:
        return

    # Build batter -> most common side mapping
    side_counts = (
        pitches.groupby(["Batter", "BatterSide"])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .drop_duplicates(subset=["Batter"], keep="first")
    )
    batter_to_side = dict(zip(
        side_counts["Batter"].astype(str).str.strip(),
        side_counts["BatterSide"].astype(str).str.strip(),
    ))

    if "batsHand" not in rate_df.columns:
        rate_df["batsHand"] = "?"

    name_col = "playerFullName" if "playerFullName" in rate_df.columns else (
        "fullName" if "fullName" in rate_df.columns else None
    )
    if name_col is None:
        return

    for idx, row in rate_df.iterrows():
        current = row.get("batsHand", "?")
        if current not in [None, "", "?"] and not (isinstance(current, float) and pd.isna(current)):
            continue
        name = str(row[name_col]).strip()
        side = batter_to_side.get(name)
        if side:
            rate_df.at[idx, "batsHand"] = side


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
    the row with more playing time — PA for hitting tables, IP/BF for pitching.
    This ensures injured players (e.g., hurt in 2025) use their fuller 2024 data.
    """
    # Initialize structure
    all_tables = {
        "hitting": _HITTING_TABLES,
        "pitching": _PITCHING_TABLES,
        "catching": _CATCHING_TABLES,
        "defense": _DEFENSE_TABLES,
    }
    merged: Dict[str, Any] = {}

    # Determine the "volume" column for dedup preference
    _VOLUME_COLS = {
        "hitting": "PA",
        "pitching": "IP",
        "catching": "PA",
        "defense": "Inn",
    }
    _VOLUME_FALLBACKS = ["BF", "PA", "G", "IP", "Inn"]

    for group_name, table_names in all_tables.items():
        merged[group_name] = {}
        vol_col = _VOLUME_COLS.get(group_name, "PA")

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

            # Deduplicate: keep the row with more playing time per player
            name_col = None
            for c in _NAME_COL_CANDIDATES:
                if c in combined.columns:
                    name_col = c
                    break
            if name_col:
                # Find best volume column available in this table
                best_vol = None
                for vc in [vol_col] + _VOLUME_FALLBACKS:
                    if vc in combined.columns:
                        best_vol = vc
                        break

                if best_vol:
                    combined[best_vol] = pd.to_numeric(combined[best_vol], errors="coerce").fillna(0)
                    combined = combined.sort_values(best_vol, ascending=True)
                    combined = combined.drop_duplicates(subset=[name_col], keep="last")
                else:
                    # No volume column — keep last (most recent season)
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

    # Report which roster players were found / missing
    found_hitters = set()
    found_pitchers = set()
    for tbl_name, tbl in [("rate", combined.get("hitting", {}).get("rate", pd.DataFrame())),
                          ("counting", combined.get("hitting", {}).get("counting", pd.DataFrame()))]:
        if isinstance(tbl, pd.DataFrame) and not tbl.empty and "playerFullName" in tbl.columns:
            found_hitters.update(tbl["playerFullName"].dropna().astype(str).str.strip())
    for tbl in [combined.get("pitching", {}).get("traditional", pd.DataFrame()),
                combined.get("pitching", {}).get("rate", pd.DataFrame())]:
        if isinstance(tbl, pd.DataFrame) and not tbl.empty and "playerFullName" in tbl.columns:
            found_pitchers.update(tbl["playerFullName"].dropna().astype(str).str.strip())
    found_all = found_hitters | found_pitchers
    missing = roster_display - found_all
    _log(f"  Found {len(found_hitters)} hitters, {len(found_pitchers)} pitchers")
    if missing:
        _log(f"  NOT FOUND in either year: {sorted(missing)}")

    # Stamp newestTeamName on all DataFrames so _tm_team() filtering works
    # when the scouting page passes team="Bryant (2024-25 Combined)"
    _COMBINED_LABEL = "Bryant (2024-25 Combined)"
    for group_name in ("hitting", "pitching", "catching", "defense"):
        group = combined.get(group_name, {})
        for table_name, df in group.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                if "newestTeamName" in df.columns:
                    df["newestTeamName"] = _COMBINED_LABEL
                elif "mostRecentTeamName" in df.columns:
                    df["mostRecentTeamName"] = _COMBINED_LABEL
                    df["newestTeamName"] = _COMBINED_LABEL

    # ── Enrich with pitch-level data (hole scores, bats, count-zone metrics) ──
    # Fetch GamePitchesTrackman from TrueMedia for each season.
    # 1. Bryant team pitches (covers all returning players)
    # 2. Transfer players' previous school pitches (covers transfers)
    _log("Fetching pitch-level data for hole scores...")
    from data.truemedia_api import fetch_team_all_pitches_trackman
    from decision_engine.data.opponent_pack import _compute_and_store_hole_scores

    all_pitch_dfs = []      # hitter-side pitches (roster player is Batter)
    all_pitcher_dfs = []    # pitcher-side pitches (roster player is Pitcher)
    # Track which (school, season) combos we've already fetched to avoid duplicates
    _fetched: Set[Tuple[str, int]] = set()

    for season in sorted(seasons):
        teams_df = fetch_all_teams(season_year=season)
        if teams_df is None or teams_df.empty:
            continue

        # Build list of (school_search_name, player_set_for_filter) to fetch
        fetch_targets: List[Tuple[str, Set[str]]] = []

        # Always fetch Bryant (covers returning players + transfers who already played for Bryant)
        fetch_targets.append((BRYANT_TEAM_NAME, roster_display))

        # Also fetch each transfer's previous school to get their pitch data there
        for roster_name, prev_school in BRYANT_TRANSFERS.items():
            disp = display_name(roster_name, escape_html=False)
            fetch_targets.append((prev_school, {disp}))

        for school_search, player_filter in fetch_targets:
            fetch_key = (school_search, season)
            if fetch_key in _fetched:
                continue

            result = _find_team_id(teams_df, school_search)
            if result is None:
                continue
            team_id_season, team_full = result
            _fetched.add(fetch_key)

            try:
                pitches = fetch_team_all_pitches_trackman(team_id_season, season)
                if pitches is not None and not pitches.empty:
                    # Filter HITTER-side: pitches where a roster player is batting
                    if "Batter" in pitches.columns:
                        exact_f, norm_f, fl_f, il_f = _build_name_matcher(player_filter)
                        mask = pitches["Batter"].apply(
                            lambda x, e=exact_f, n=norm_f, f=fl_f, i=il_f: _name_matches(x, e, n, f, i)
                        )
                        filtered_pitches = pitches[mask]
                    else:
                        filtered_pitches = pitches

                    if len(filtered_pitches) > 0:
                        _log(f"  {team_full} {season}: {len(filtered_pitches)} hitter pitches (of {len(pitches)} total)")
                        all_pitch_dfs.append(filtered_pitches)

                    # Filter PITCHER-side: pitches where a roster player is pitching
                    if "Pitcher" in pitches.columns:
                        p_mask = pitches["Pitcher"].apply(
                            lambda x, e=exact_f, n=norm_f, f=fl_f, i=il_f: _name_matches(x, e, n, f, i)
                        )
                        pitcher_pitches = pitches[p_mask]
                        if len(pitcher_pitches) > 0:
                            _log(f"  {team_full} {season}: {len(pitcher_pitches)} pitcher pitches")
                            all_pitcher_dfs.append(pitcher_pitches)
            except Exception as e:
                _log(f"  {school_search} {season}: pitch fetch failed: {e}")

    if all_pitch_dfs:
        roster_pitches = pd.concat(all_pitch_dfs, ignore_index=True)

        # Deduplicate: same pitch may appear if a transfer played Bryant (fetched via both paths)
        # TrueMedia uses trackmanPitchUID as unique pitch identifier.
        # Only dedup rows that have a non-null UID (null UIDs are kept as-is).
        uid_col = "trackmanPitchUID"
        if uid_col in roster_pitches.columns and "Batter" in roster_pitches.columns:
            has_uid = roster_pitches[uid_col].notna()
            with_uid = roster_pitches[has_uid]
            without_uid = roster_pitches[~has_uid]
            before = len(with_uid)
            with_uid = with_uid.drop_duplicates(subset=[uid_col, "Batter"], keep="first")
            roster_pitches = pd.concat([with_uid, without_uid], ignore_index=True)
            if len(with_uid) < before:
                _log(f"  Deduped: {before} -> {len(with_uid)} rows with UID")

        _log(f"  Total roster pitches: {len(roster_pitches)}")

        # Infer bats from BatterSide for hitters missing it in rate table
        rate_df = combined.get("hitting", {}).get("rate", pd.DataFrame())
        if not rate_df.empty and "BatterSide" in roster_pitches.columns:
            _enrich_bats_from_pitches(rate_df, roster_pitches)

        # Compute hole scores + count-zone metrics
        if len(roster_pitches) >= 30:
            _compute_and_store_hole_scores(combined, roster_pitches)
            hs = combined.get("hitting", {}).get("hole_scores")
            _log(f"  Hole scores: {len(hs)} rows" if hs is not None else "  No hole scores computed")

    # Cache it
    _log("Saving combined pack to cache...")
    save_opponent_pack(
        combined,
        team_id=BRYANT_COMBINED_TEAM_ID,
        team_name="Bryant (2024-25 Combined)",
        season_year=2026,
    )

    # Also cache pitch-level data so the scouting page can build zone heatmaps
    if all_pitch_dfs and len(roster_pitches) > 0:
        _pitches_path = _bryant_pitches_path()
        os.makedirs(os.path.dirname(_pitches_path), exist_ok=True)
        roster_pitches.to_parquet(_pitches_path, index=False)
        _log(f"  Saved {len(roster_pitches)} hitter pitches to cache")

    # Cache pitcher-side pitches (for pitcher analysis, baserunning, pickoffs)
    if all_pitcher_dfs:
        pitcher_pitches_all = pd.concat(all_pitcher_dfs, ignore_index=True)
        uid_col = "trackmanPitchUID"
        if uid_col in pitcher_pitches_all.columns:
            pitcher_pitches_all = pitcher_pitches_all.drop_duplicates(subset=[uid_col], keep="first")
        _pp_path = _bryant_pitcher_pitches_path()
        os.makedirs(os.path.dirname(_pp_path), exist_ok=True)
        pitcher_pitches_all.to_parquet(_pp_path, index=False)
        _log(f"  Saved {len(pitcher_pitches_all)} pitcher pitches to cache")

    _log("Done!")
    return combined


def _bryant_pitches_path() -> str:
    """Path for cached Bryant combined hitter-side pitch-level data."""
    from decision_engine.data.opponent_pack import _pack_dir
    return os.path.join(_pack_dir(BRYANT_COMBINED_TEAM_ID, 2026), "pitches.parquet")


def _bryant_pitcher_pitches_path() -> str:
    """Path for cached Bryant combined pitcher-side pitch-level data."""
    from decision_engine.data.opponent_pack import _pack_dir
    return os.path.join(_pack_dir(BRYANT_COMBINED_TEAM_ID, 2026), "pitcher_pitches.parquet")


def load_bryant_combined_pack() -> Optional[Dict[str, Any]]:
    """Load the cached Bryant combined pack, or None if not built yet."""
    pack = load_opponent_pack(BRYANT_COMBINED_TEAM_ID, season_year=2026)
    if pack is None:
        return None
    # Ensure newestTeamName is stamped correctly (older caches may have original school names)
    _COMBINED_LABEL = "Bryant (2024-25 Combined)"
    for group_name in ("hitting", "pitching", "catching", "defense"):
        for df in pack.get(group_name, {}).values():
            if isinstance(df, pd.DataFrame) and not df.empty and "newestTeamName" in df.columns:
                df["newestTeamName"] = _COMBINED_LABEL
    return pack


def load_bryant_pitches() -> pd.DataFrame:
    """Load cached hitter-side pitch-level data for the Bryant combined pack."""
    fp = _bryant_pitches_path()
    if os.path.exists(fp):
        try:
            return pd.read_parquet(fp)
        except Exception:
            pass
    return pd.DataFrame()


def load_bryant_pitcher_pitches() -> pd.DataFrame:
    """Load cached pitcher-side pitch-level data for the Bryant combined pack."""
    fp = _bryant_pitcher_pitches_path()
    if os.path.exists(fp):
        try:
            return pd.read_parquet(fp)
        except Exception:
            pass
    return pd.DataFrame()
