#!/usr/bin/env python3
"""Pre-fetch and cache the Bryant 2026 combined opponent pack.

Fetches 2024+2025 data for returning Bryant players, and previous-school
data for transfers, then merges into a single cached pack.
"""
from __future__ import annotations

import argparse

from data.bryant_combined import build_bryant_combined_pack


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Bryant 2026 combined opponent pack")
    parser.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025],
                        help="Seasons to include (default: 2024 2025)")
    parser.add_argument("--refresh", action="store_true", help="Force re-fetch from API")
    args = parser.parse_args()

    def _log(msg):
        print(msg)

    pack = build_bryant_combined_pack(
        refresh=args.refresh,
        seasons=sorted(args.seasons),
        progress_callback=_log,
    )

    h_rate = pack.get("hitting", {}).get("rate")
    p_trad = pack.get("pitching", {}).get("traditional")
    n_h = len(h_rate) if hasattr(h_rate, "__len__") and not getattr(h_rate, "empty", True) else 0
    n_p = len(p_trad) if hasattr(p_trad, "__len__") and not getattr(p_trad, "empty", True) else 0
    print(f"\nResult: {n_h} hitters, {n_p} pitchers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
