from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import duckdb
import numpy as np
import pandas as pd

from config import (
    CACHE_DIR,
    PARQUET_PATH,
    ZONE_SIDE,
    ZONE_HEIGHT_BOT,
    ZONE_HEIGHT_TOP,
)


def _parquet_fingerprint(path: str) -> Dict[str, Optional[float]]:
    try:
        return {"path": path, "mtime": os.path.getmtime(path), "size": os.path.getsize(path)}
    except OSError:
        return {"path": path, "mtime": None, "size": None}


def _cache_path() -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, "decision_engine_pitch_priors.json")


@dataclass(frozen=True)
class PitchPriors:
    overall: Dict[str, float]
    by_pitch_type: Dict[str, Dict[str, float]]


def compute_pitch_priors(parquet_path: str = PARQUET_PATH) -> PitchPriors:
    """Compute simple D1 pitch-type priors from the local Trackman parquet."""
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    swing_calls = "('StrikeSwinging','FoulBall','FoulBallNotFieldable','FoulBallFieldable','InPlay')"
    has_loc = "PlateLocSide IS NOT NULL AND PlateLocHeight IS NOT NULL"
    in_zone = f"ABS(PlateLocSide) <= {ZONE_SIDE} AND PlateLocHeight BETWEEN {ZONE_HEIGHT_BOT} AND {ZONE_HEIGHT_TOP}"
    out_zone = f"({has_loc}) AND NOT ({in_zone})"
    barrel_cond = (
        "ExitSpeed>=98 AND Angle IS NOT NULL AND "
        "Angle >= GREATEST(26 - 2*(ExitSpeed-98), 8) AND "
        "Angle <= LEAST(30 + 3*(ExitSpeed-98), 50)"
    )

    # Normalize pitch types to match `config.normalize_pitch_types()` / arsenal builder output.
    pt_norm = (
        "CASE "
        "WHEN TaggedPitchType IN ('Undefined','Other','Knuckleball') THEN NULL "
        "WHEN TaggedPitchType = 'FourSeamFastBall' THEN 'Fastball' "
        "WHEN TaggedPitchType IN ('OneSeamFastBall','TwoSeamFastBall') THEN 'Sinker' "
        "WHEN TaggedPitchType = 'ChangeUp' THEN 'Changeup' "
        "ELSE TaggedPitchType END"
    )

    con = duckdb.connect(database=":memory:")
    df = con.execute(
        f"""
        WITH base AS (
          SELECT
            {pt_norm} AS pitch_type,
            PitchCall,
            PlateLocSide,
            PlateLocHeight,
            ExitSpeed,
            Angle
          FROM read_parquet('{parquet_path.replace("'", "''")}')
          WHERE PitchCall IS NULL OR PitchCall != 'Undefined'
        )
        SELECT
          pitch_type,
          COUNT(*) AS n_pitches,
          SUM(CASE WHEN PitchCall IN {swing_calls} THEN 1 ELSE 0 END) AS n_swings,
          SUM(CASE WHEN PitchCall='StrikeSwinging' THEN 1 ELSE 0 END) AS n_whiffs,
          SUM(CASE WHEN PitchCall IN ('StrikeCalled','StrikeSwinging') THEN 1 ELSE 0 END) AS n_csw,
          SUM(CASE WHEN {out_zone} THEN 1 ELSE 0 END) AS n_oz_pitches,
          SUM(CASE WHEN {out_zone} AND PitchCall IN {swing_calls} THEN 1 ELSE 0 END) AS n_oz_swings,
          SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN 1 ELSE 0 END) AS n_inplay_ev,
          AVG(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN ExitSpeed END) AS ev_against,
          SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND {barrel_cond} THEN 1 ELSE 0 END) AS n_barrels
        FROM base
        WHERE pitch_type IS NOT NULL
        GROUP BY pitch_type
        ORDER BY n_pitches DESC
        """
    ).fetchdf()

    if df.empty:
        empty = {
            "whiff_pct": float("nan"),
            "csw_pct": float("nan"),
            "chase_pct": float("nan"),
            "ev_against": float("nan"),
            "barrel_pct_against": float("nan"),
        }
        return PitchPriors(overall=empty, by_pitch_type={})

    df["whiff_pct"] = np.where(df["n_swings"] > 0, df["n_whiffs"] / df["n_swings"] * 100, np.nan)
    df["csw_pct"] = np.where(df["n_pitches"] > 0, df["n_csw"] / df["n_pitches"] * 100, np.nan)
    df["chase_pct"] = np.where(df["n_oz_pitches"] > 0, df["n_oz_swings"] / df["n_oz_pitches"] * 100, np.nan)
    df["barrel_pct_against"] = np.where(df["n_inplay_ev"] > 0, df["n_barrels"] / df["n_inplay_ev"] * 100, np.nan)

    by_pitch_type = {}
    for _, row in df.iterrows():
        pt = str(row["pitch_type"])
        by_pitch_type[pt] = {
            "whiff_pct": float(row["whiff_pct"]) if pd.notna(row["whiff_pct"]) else float("nan"),
            "csw_pct": float(row["csw_pct"]) if pd.notna(row["csw_pct"]) else float("nan"),
            "chase_pct": float(row["chase_pct"]) if pd.notna(row["chase_pct"]) else float("nan"),
            "ev_against": float(row["ev_against"]) if pd.notna(row["ev_against"]) else float("nan"),
            "barrel_pct_against": float(row["barrel_pct_against"]) if pd.notna(row["barrel_pct_against"]) else float("nan"),
            "n_pitches": int(row["n_pitches"]),
        }

    # Overall priors should be aggregated across all pitches (not an unweighted average
    # of pitch types, which overweights rare pitch types).
    total_pitches = float(df["n_pitches"].sum())
    total_swings = float(df["n_swings"].sum())
    total_whiffs = float(df["n_whiffs"].sum())
    total_csw = float(df["n_csw"].sum())
    total_oz_pitches = float(df["n_oz_pitches"].sum())
    total_oz_swings = float(df["n_oz_swings"].sum())
    total_inplay_ev = float(df["n_inplay_ev"].sum())
    total_barrels = float(df["n_barrels"].sum())

    # Weighted EV against (only where EV exists).
    ev_mask = (df["n_inplay_ev"] > 0) & df["ev_against"].notna()
    ev_weight = float(df.loc[ev_mask, "n_inplay_ev"].sum())
    ev_weighted = float(
        (df.loc[ev_mask, "ev_against"] * df.loc[ev_mask, "n_inplay_ev"]).sum() / ev_weight
    ) if ev_weight > 0 else float("nan")

    overall = {
        "whiff_pct": float(total_whiffs / total_swings * 100) if total_swings > 0 else float("nan"),
        "csw_pct": float(total_csw / total_pitches * 100) if total_pitches > 0 else float("nan"),
        "chase_pct": float(total_oz_swings / total_oz_pitches * 100) if total_oz_pitches > 0 else float("nan"),
        "ev_against": ev_weighted,
        "barrel_pct_against": float(total_barrels / total_inplay_ev * 100) if total_inplay_ev > 0 else float("nan"),
    }
    return PitchPriors(overall=overall, by_pitch_type=by_pitch_type)


def load_pitch_priors(parquet_path: str = PARQUET_PATH, force_refresh: bool = False) -> PitchPriors:
    """Load pitch priors from disk cache, recomputing if parquet changed."""
    cache_path = _cache_path()
    fp = _parquet_fingerprint(parquet_path)

    if not force_refresh and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                blob = json.load(f)
            if blob.get("fingerprint") == fp and "priors" in blob:
                pri = blob["priors"]
                return PitchPriors(
                    overall=pri.get("overall", {}),
                    by_pitch_type=pri.get("by_pitch_type", {}),
                )
        except Exception:
            pass

    priors = compute_pitch_priors(parquet_path=parquet_path)
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"fingerprint": fp, "priors": {"overall": priors.overall, "by_pitch_type": priors.by_pitch_type}}, f)
    except Exception:
        pass
    return priors
