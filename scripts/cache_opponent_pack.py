#!/usr/bin/env python3
"""Pre-fetch and cache a TrueMedia opponent pack to disk for offline use."""

from __future__ import annotations

import argparse

from decision_engine.data.opponent_pack import load_or_build_opponent_pack


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache a TrueMedia opponent pack to disk")
    parser.add_argument("--season", type=int, default=2026, help="Season year")
    parser.add_argument("--team-id", required=True, help="TrueMedia teamId")
    parser.add_argument("--team-name", required=True, help="Team name (display + newestTeamName)")
    parser.add_argument("--refresh", action="store_true", help="Force refresh from API")
    args = parser.parse_args()

    pack = load_or_build_opponent_pack(
        team_id=args.team_id,
        team_name=args.team_name,
        season_year=args.season,
        refresh=args.refresh,
    )
    n_hit = len(pack.get("hitting", {}).get("rate", []) or [])
    n_pit = len(pack.get("pitching", {}).get("traditional", []) or [])
    print(f"Cached opponent pack: {args.team_name} ({args.team_id}) season {args.season}")
    print(f"Hitting rows: {n_hit:,}")
    print(f"Pitching rows: {n_pit:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

