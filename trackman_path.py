from __future__ import annotations

from pathlib import Path


def resolve_latest_trackman_parquet(base_dir: str | Path) -> str:
    """Return the newest local Trackman parquet in the project root."""
    root = Path(base_dir)
    candidates = [p for p in root.glob("all_trackman*.parquet") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No Trackman parquet found in {root}")

    candidates.sort(
        key=lambda p: (
            p.stat().st_mtime,
            "fixed" in p.name.lower(),
            p.name,
        ),
        reverse=True,
    )
    return str(candidates[0])
