"""
Run Expectancy Calibration — RE24, pitch outcome probs, contact RV, and ΔRE.

Replaces arbitrary 0-100 composite scores with ΔRE (change in run expectancy)
so that pitch values are comparable across situations.

Uses the same cache-with-fingerprint pattern as ``count_calibration.py`` and
``historical_calibration.py``.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import numpy as np

from config import CACHE_DIR, PARQUET_PATH


# ── Pitch-type normalization SQL (shared) ─────────────────────────────────────
_PT_NORM = (
    "CASE "
    "WHEN TaggedPitchType IN ('Undefined','Other','Knuckleball') THEN NULL "
    "WHEN TaggedPitchType = 'FourSeamFastBall' THEN 'Fastball' "
    "WHEN TaggedPitchType IN ('OneSeamFastBall','TwoSeamFastBall') THEN 'Sinker' "
    "WHEN TaggedPitchType = 'ChangeUp' THEN 'Changeup' "
    "WHEN TaggedPitchType = 'KnuckleCurve' THEN 'Knuckle Curve' "
    "ELSE TaggedPitchType END"
)

# Barrel definition
_BARREL_COND = (
    "ExitSpeed >= 98 AND Angle IS NOT NULL AND "
    "Angle >= GREATEST(26 - 2*(ExitSpeed - 98), 8) AND "
    "Angle <= LEAST(30 + 3*(ExitSpeed - 98), 50)"
)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RE24Matrix:
    """Base-out state expected runs matrix."""
    matrix: Dict[str, float]    # "(on1b,on2b,on3b,outs)" -> expected_runs
    n_obs: Dict[str, int]       # same key -> observation count


@dataclass(frozen=True)
class PitchOutcomeProbs:
    """Pitch outcome probability distribution for a single (pitch_type, count)."""
    p_ball: float
    p_called_strike: float
    p_swinging_strike: float
    p_foul: float
    p_in_play: float
    p_hbp: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "p_ball": self.p_ball,
            "p_called_strike": self.p_called_strike,
            "p_swinging_strike": self.p_swinging_strike,
            "p_foul": self.p_foul,
            "p_in_play": self.p_in_play,
            "p_hbp": self.p_hbp,
        }


@dataclass
class RunExpectancyCalibration:
    """Full RE calibration result."""
    re24: Dict[str, float]                                    # base-out key -> expected runs
    re24_n: Dict[str, int]                                    # base-out key -> n observations
    outcome_probs: Dict[str, Dict[str, Dict[str, float]]]    # pitch -> count -> probs dict
    contact_rv: Dict[str, float]                              # pitch_type -> expected wOBA on contact
    count_rv: Dict[str, float]                                # "b-s" -> count run value


# ── RE24 state key helpers ────────────────────────────────────────────────────

def _bo_key(on1b: int, on2b: int, on3b: int, outs: int) -> str:
    return f"{on1b},{on2b},{on3b},{outs}"


def _parse_bo_key(key: str) -> Tuple[int, int, int, int]:
    parts = key.split(",")
    return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])


# ── 1A. RE24 Matrix computation ──────────────────────────────────────────────

def _advance_runners_single(on1b: int, on2b: int, on3b: int) -> Tuple[int, int, int, int]:
    """Single: R3 scores, R2->3B, R1->2B, batter->1B."""
    runs = on3b
    new_3b = on2b
    new_2b = on1b
    new_1b = 1
    return new_1b, new_2b, new_3b, runs


def _advance_runners_double(on1b: int, on2b: int, on3b: int) -> Tuple[int, int, int, int]:
    """Double: R3+R2 score, R1->3B, batter->2B."""
    runs = on3b + on2b
    new_3b = on1b
    new_2b = 1
    new_1b = 0
    return new_1b, new_2b, new_3b, runs


def _advance_runners_triple(on1b: int, on2b: int, on3b: int) -> Tuple[int, int, int, int]:
    """Triple: all score, batter->3B."""
    runs = on1b + on2b + on3b
    return 0, 0, 1, runs


def _advance_runners_hr(on1b: int, on2b: int, on3b: int) -> Tuple[int, int, int, int]:
    """HR: all + batter score."""
    runs = on1b + on2b + on3b + 1
    return 0, 0, 0, runs


def _advance_runners_walk(on1b: int, on2b: int, on3b: int) -> Tuple[int, int, int, int]:
    """BB/HBP: forced advances only."""
    runs = 0
    if on1b and on2b and on3b:
        runs = 1  # R3 forced home
    new_3b = on3b if not (on1b and on2b) else (1 if on2b else on3b)
    if on1b and on2b:
        new_3b = 1  # R2 forced to 3B
    elif not on1b:
        new_3b = on3b
    else:
        # R1 occupied, R2 empty -> R1->2B, no force at 3B
        new_3b = on3b

    # More precise forced-advance logic:
    if on1b:
        new_2b = 1
        if on2b:
            new_3b = 1
            if on3b:
                runs = 1
        else:
            new_3b = on3b
    else:
        new_2b = on2b
        new_3b = on3b
    new_1b = 1
    return new_1b, new_2b, new_3b, runs


def _reconstruct_half_inning(pa_sequence: List[Dict]) -> List[Dict]:
    """Replay a half-inning's PA outcomes to reconstruct base-out states.

    Each PA dict must have keys: 'outcome', 'runs_scored'.
    Returns the same list with added keys: 'on1b', 'on2b', 'on3b', 'outs_before'.
    """
    on1b, on2b, on3b = 0, 0, 0
    outs = 0
    results = []

    for pa in pa_sequence:
        if outs >= 3:
            break
        pa_out = dict(pa)
        pa_out["on1b"] = on1b
        pa_out["on2b"] = on2b
        pa_out["on3b"] = on3b
        pa_out["outs_before"] = outs
        results.append(pa_out)

        outcome = pa.get("outcome", "Out")

        if outcome == "Single" or outcome == "Error":
            on1b, on2b, on3b, _ = _advance_runners_single(on1b, on2b, on3b)
        elif outcome == "Double":
            on1b, on2b, on3b, _ = _advance_runners_double(on1b, on2b, on3b)
        elif outcome == "Triple":
            on1b, on2b, on3b, _ = _advance_runners_triple(on1b, on2b, on3b)
        elif outcome == "HomeRun":
            on1b, on2b, on3b, _ = _advance_runners_hr(on1b, on2b, on3b)
        elif outcome in ("Walk", "HitByPitch"):
            on1b, on2b, on3b, _ = _advance_runners_walk(on1b, on2b, on3b)
        elif outcome == "SacFly":
            outs += 1
            # R3 scores on sac fly
            if on3b:
                on3b = 0
        elif outcome in ("Strikeout", "Out", "FieldersChoice", "Sacrifice"):
            outs += 1
        # else: unknown, treat as out
        else:
            outs += 1

    return results


def compute_re24_matrix(parquet_path: str = PARQUET_PATH) -> RE24Matrix:
    """Compute RE24 base-out state run expectancy from Trackman play-by-play."""
    if not os.path.exists(parquet_path):
        return _fallback_re24()

    pq = parquet_path.replace("'", "''")
    con = duckdb.connect(database=":memory:")

    # Get PA-level outcomes grouped by half-inning
    rows = con.execute(f"""
        WITH pa_data AS (
            SELECT
                GameID,
                CAST(Inning AS INTEGER) AS Inning,
                BatterTeam,
                CAST(PAofInning AS INTEGER) AS PAofInning,
                Batter,
                HomeTeam,
                AwayTeam,
                CASE
                    WHEN KorBB = 'Strikeout' THEN 'Strikeout'
                    WHEN KorBB = 'Walk' THEN 'Walk'
                    WHEN PitchCall = 'HitByPitch' THEN 'HitByPitch'
                    WHEN PitchCall = 'InPlay' AND PlayResult = 'Single' THEN 'Single'
                    WHEN PitchCall = 'InPlay' AND PlayResult = 'Double' THEN 'Double'
                    WHEN PitchCall = 'InPlay' AND PlayResult = 'Triple' THEN 'Triple'
                    WHEN PitchCall = 'InPlay' AND PlayResult = 'HomeRun' THEN 'HomeRun'
                    WHEN PitchCall = 'InPlay' AND PlayResult = 'Error' THEN 'Error'
                    WHEN PitchCall = 'InPlay' AND PlayResult = 'Sacrifice' THEN 'Sacrifice'
                    WHEN PitchCall = 'InPlay' AND PlayResult = 'FieldersChoice' THEN 'FieldersChoice'
                    WHEN PitchCall = 'InPlay' AND PlayResult IN ('Out') THEN 'Out'
                    ELSE NULL
                END AS outcome,
                COALESCE(CAST(RunsScored AS INTEGER), 0) AS runs_scored
            FROM read_parquet('{pq}')
            WHERE PitchCall IS NOT NULL AND PitchCall != 'Undefined'
              AND HomeTeam IS NOT NULL AND AwayTeam IS NOT NULL
              AND HomeTeam != AwayTeam
              AND (PitchCall = 'InPlay' OR KorBB IN ('Strikeout','Walk') OR PitchCall = 'HitByPitch')
        )
        SELECT
            GameID, Inning, BatterTeam, PAofInning, outcome, runs_scored
        FROM pa_data
        WHERE outcome IS NOT NULL
        ORDER BY GameID, Inning, BatterTeam, PAofInning
    """).fetchall()
    con.close()

    # Group by half-inning
    half_innings: Dict[str, List[Dict]] = {}
    for game_id, inning, batter_team, pa_of_inning, outcome, runs_scored in rows:
        key = f"{game_id}_{inning}_{batter_team}"
        if key not in half_innings:
            half_innings[key] = []
        half_innings[key].append({
            "pa_of_inning": pa_of_inning,
            "outcome": outcome,
            "runs_scored": int(runs_scored),
        })

    # Reconstruct base-out states and compute runs-rest-of-inning
    state_runs: Dict[str, List[float]] = {}  # bo_key -> list of runs_rest

    for hi_key, pa_list in half_innings.items():
        # Sort by PA order
        pa_list.sort(key=lambda x: x["pa_of_inning"])

        # Reconstruct
        reconstructed = _reconstruct_half_inning(pa_list)

        # Total runs in the half-inning
        total_runs = sum(pa["runs_scored"] for pa in pa_list)

        # For each PA, compute runs from this point onward
        cumulative = 0
        for i, pa in enumerate(reconstructed):
            runs_before = cumulative
            # runs_rest = total_runs - runs_scored_before_this_PA
            runs_rest = total_runs - runs_before

            bo = _bo_key(pa["on1b"], pa["on2b"], pa["on3b"], pa["outs_before"])
            if bo not in state_runs:
                state_runs[bo] = []
            state_runs[bo].append(float(runs_rest))

            cumulative += pa_list[i]["runs_scored"]

    # Compute averages
    matrix: Dict[str, float] = {}
    n_obs: Dict[str, int] = {}
    for bo, runs_list in state_runs.items():
        matrix[bo] = round(float(np.mean(runs_list)), 4)
        n_obs[bo] = len(runs_list)

    # Fill in any missing states with fallback values
    fallback = _fallback_re24()
    for outs in range(3):
        for b1 in range(2):
            for b2 in range(2):
                for b3 in range(2):
                    key = _bo_key(b1, b2, b3, outs)
                    if key not in matrix or n_obs.get(key, 0) < 20:
                        matrix[key] = fallback.matrix.get(key, 0.0)
                        n_obs[key] = n_obs.get(key, 0)

    # 3 outs = 0 runs (inning over)
    for b1 in range(2):
        for b2 in range(2):
            for b3 in range(2):
                key = _bo_key(b1, b2, b3, 3)
                matrix[key] = 0.0
                n_obs[key] = 0

    return RE24Matrix(matrix=matrix, n_obs=n_obs)


def _fallback_re24() -> RE24Matrix:
    """Hardcoded D1-scaled RE24 (MLB × ~1.15 for D1 run environment)."""
    # MLB RE24 averages × 1.15 scaling for higher-scoring D1 environment
    mlb_base = {
        # (on1b, on2b, on3b, outs): expected_runs
        (0, 0, 0, 0): 0.555, (0, 0, 0, 1): 0.297, (0, 0, 0, 2): 0.117,
        (1, 0, 0, 0): 0.953, (1, 0, 0, 1): 0.573, (1, 0, 0, 2): 0.251,
        (0, 1, 0, 0): 1.189, (0, 1, 0, 1): 0.725, (0, 1, 0, 2): 0.344,
        (0, 0, 1, 0): 1.482, (0, 0, 1, 1): 0.983, (0, 0, 1, 2): 0.387,
        (1, 1, 0, 0): 1.573, (1, 1, 0, 1): 0.971, (1, 1, 0, 2): 0.466,
        (1, 0, 1, 0): 1.904, (1, 0, 1, 1): 1.243, (1, 0, 1, 2): 0.538,
        (0, 1, 1, 0): 2.052, (0, 1, 1, 1): 1.467, (0, 1, 1, 2): 0.601,
        (1, 1, 1, 0): 2.417, (1, 1, 1, 1): 1.650, (1, 1, 1, 2): 0.815,
    }
    d1_scale = 1.15
    matrix = {}
    n_obs = {}
    for (b1, b2, b3, o), rv in mlb_base.items():
        key = _bo_key(b1, b2, b3, o)
        matrix[key] = round(rv * d1_scale, 4)
        n_obs[key] = 0
    # 3 outs = 0
    for b1 in range(2):
        for b2 in range(2):
            for b3 in range(2):
                key = _bo_key(b1, b2, b3, 3)
                matrix[key] = 0.0
                n_obs[key] = 0
    return RE24Matrix(matrix=matrix, n_obs=n_obs)


# ── 1B. Pitch Outcome Probability Table ───────────────────────────────────────

def compute_pitch_outcome_probs(
    parquet_path: str = PARQUET_PATH,
) -> Dict[str, Dict[str, PitchOutcomeProbs]]:
    """Compute P(outcome | pitch_type, count) from Trackman data.

    Returns: {pitch_type: {count_str: PitchOutcomeProbs}}
    """
    if not os.path.exists(parquet_path):
        return _fallback_outcome_probs_table()

    pq = parquet_path.replace("'", "''")
    con = duckdb.connect(database=":memory:")

    rows = con.execute(f"""
        SELECT
            {_PT_NORM} AS pitch_type,
            CAST(Balls AS INTEGER) AS balls,
            CAST(Strikes AS INTEGER) AS strikes,
            PitchCall,
            COUNT(*) AS n
        FROM read_parquet('{pq}')
        WHERE PitchCall IS NOT NULL AND PitchCall != 'Undefined'
          AND ({_PT_NORM}) IS NOT NULL
          AND Balls BETWEEN 0 AND 3
          AND Strikes BETWEEN 0 AND 2
        GROUP BY pitch_type, balls, strikes, PitchCall
    """).fetchall()
    con.close()

    # Aggregate into (pitch_type, count) -> {call: n}
    agg: Dict[str, Dict[str, Dict[str, int]]] = {}
    for pt, b, s, call, n in rows:
        count_key = f"{b}-{s}"
        if pt not in agg:
            agg[pt] = {}
        if count_key not in agg[pt]:
            agg[pt][count_key] = {}
        agg[pt][count_key][call] = agg[pt][count_key].get(call, 0) + int(n)

    # Compute pitch-type overall averages for shrinkage
    pt_overall: Dict[str, Dict[str, int]] = {}
    for pt, counts in agg.items():
        pt_overall[pt] = {}
        for count_key, calls in counts.items():
            for call, n in calls.items():
                pt_overall[pt][call] = pt_overall[pt].get(call, 0) + n

    n_prior_equiv = 200

    result: Dict[str, Dict[str, PitchOutcomeProbs]] = {}
    for pt, counts in agg.items():
        result[pt] = {}
        # Pitch-type overall rates for shrinkage
        total_overall = sum(pt_overall[pt].values())
        if total_overall == 0:
            continue
        overall_rates = {c: n / total_overall for c, n in pt_overall[pt].items()}

        for count_key, calls in counts.items():
            total = sum(calls.values())
            if total < 10:
                continue

            # Raw rates
            raw = {}
            for c in ["BallCalled", "StrikeCalled", "StrikeSwinging",
                       "FoulBall", "FoulBallNotFieldable", "FoulBallFieldable",
                       "InPlay", "HitByPitch"]:
                raw[c] = calls.get(c, 0) / total

            # Shrink toward pitch-type overall (n_prior_equiv=200)
            w = total / (total + n_prior_equiv)

            def _shrink(raw_rate, call_name):
                prior_rate = overall_rates.get(call_name, 0.0)
                return w * raw_rate + (1.0 - w) * prior_rate

            p_ball = _shrink(raw.get("BallCalled", 0), "BallCalled")
            p_cs = _shrink(raw.get("StrikeCalled", 0), "StrikeCalled")
            p_ss = _shrink(raw.get("StrikeSwinging", 0), "StrikeSwinging")
            p_foul = _shrink(
                raw.get("FoulBall", 0) + raw.get("FoulBallNotFieldable", 0) + raw.get("FoulBallFieldable", 0),
                "FoulBall",
            )
            # For foul shrinkage, combine all foul types
            foul_overall = (overall_rates.get("FoulBall", 0) +
                          overall_rates.get("FoulBallNotFieldable", 0) +
                          overall_rates.get("FoulBallFieldable", 0))
            p_foul = w * (raw.get("FoulBall", 0) + raw.get("FoulBallNotFieldable", 0) +
                         raw.get("FoulBallFieldable", 0)) + (1.0 - w) * foul_overall

            p_ip = _shrink(raw.get("InPlay", 0), "InPlay")
            p_hbp = _shrink(raw.get("HitByPitch", 0), "HitByPitch")

            # Normalize to sum to 1
            total_p = p_ball + p_cs + p_ss + p_foul + p_ip + p_hbp
            if total_p > 0:
                p_ball /= total_p
                p_cs /= total_p
                p_ss /= total_p
                p_foul /= total_p
                p_ip /= total_p
                p_hbp /= total_p

            result[pt][count_key] = PitchOutcomeProbs(
                p_ball=round(p_ball, 5),
                p_called_strike=round(p_cs, 5),
                p_swinging_strike=round(p_ss, 5),
                p_foul=round(p_foul, 5),
                p_in_play=round(p_ip, 5),
                p_hbp=round(p_hbp, 5),
            )

    return result


def _fallback_outcome_probs_table() -> Dict[str, Dict[str, PitchOutcomeProbs]]:
    """Hardcoded fallback outcome probabilities for common pitch types."""
    # Representative league-average rates
    default_probs = PitchOutcomeProbs(
        p_ball=0.38, p_called_strike=0.17, p_swinging_strike=0.10,
        p_foul=0.18, p_in_play=0.16, p_hbp=0.01,
    )
    fb = PitchOutcomeProbs(
        p_ball=0.36, p_called_strike=0.19, p_swinging_strike=0.08,
        p_foul=0.20, p_in_play=0.16, p_hbp=0.01,
    )
    sl = PitchOutcomeProbs(
        p_ball=0.40, p_called_strike=0.14, p_swinging_strike=0.14,
        p_foul=0.16, p_in_play=0.15, p_hbp=0.01,
    )
    cb = PitchOutcomeProbs(
        p_ball=0.42, p_called_strike=0.16, p_swinging_strike=0.10,
        p_foul=0.14, p_in_play=0.17, p_hbp=0.01,
    )
    ch = PitchOutcomeProbs(
        p_ball=0.38, p_called_strike=0.16, p_swinging_strike=0.12,
        p_foul=0.16, p_in_play=0.17, p_hbp=0.01,
    )
    # Apply across all counts
    result = {}
    for pt, probs in [("Fastball", fb), ("Sinker", fb), ("Cutter", fb),
                       ("Slider", sl), ("Sweeper", sl), ("Curveball", cb),
                       ("Knuckle Curve", cb), ("Changeup", ch), ("Splitter", ch)]:
        result[pt] = {}
        for b in range(4):
            for s in range(3):
                result[pt][f"{b}-{s}"] = probs
    return result


# ── 1C. Contact Run Value by Pitch Type ───────────────────────────────────────

def compute_contact_rv(parquet_path: str = PARQUET_PATH) -> Dict[str, float]:
    """Compute expected wOBA on contact for each pitch type.

    Classifies BIPs into EV/LA buckets, applies outcome probs × linear weights.
    """
    if not os.path.exists(parquet_path):
        return _fallback_contact_rv()

    pq = parquet_path.replace("'", "''")
    con = duckdb.connect(database=":memory:")

    # Load linear weights from historical calibration cache
    lw = _load_linear_weights()

    rows = con.execute(f"""
        WITH bip AS (
            SELECT
                {_PT_NORM} AS pitch_type,
                ExitSpeed AS ev,
                Angle AS la,
                PlayResult,
                CASE
                    WHEN {_BARREL_COND} THEN 'Barrel'
                    WHEN ExitSpeed >= 95 AND Angle BETWEEN 25 AND 45 THEN 'HiEV_FB'
                    WHEN Angle BETWEEN 10 AND 25 AND ExitSpeed >= 85 THEN 'Hard_LD'
                    WHEN Angle < 10 THEN 'GB'
                    WHEN Angle > 45 THEN 'Popup'
                    WHEN ExitSpeed < 70 THEN 'Soft'
                    ELSE 'Medium'
                END AS bucket
            FROM read_parquet('{pq}')
            WHERE PitchCall = 'InPlay'
              AND ExitSpeed IS NOT NULL
              AND Angle IS NOT NULL
              AND PlayResult IS NOT NULL
              AND ({_PT_NORM}) IS NOT NULL
        )
        SELECT
            pitch_type,
            PlayResult,
            COUNT(*) AS n
        FROM bip
        GROUP BY pitch_type, PlayResult
    """).fetchall()
    con.close()

    # Aggregate by pitch type
    pt_results: Dict[str, Dict[str, int]] = {}
    for pt, result, n in rows:
        if pt not in pt_results:
            pt_results[pt] = {}
        pt_results[pt][result] = pt_results[pt].get(result, 0) + int(n)

    contact_rv: Dict[str, float] = {}
    for pt, results in pt_results.items():
        total = sum(results.values())
        if total < 30:
            continue
        rv = 0.0
        for outcome, n in results.items():
            pct = n / total
            weight = lw.get(outcome, 0.0)
            rv += pct * weight
        contact_rv[pt] = round(rv, 5)

    return contact_rv


def _load_linear_weights() -> Dict[str, float]:
    """Load linear weights from historical calibration cache, with fallback."""
    try:
        cache_file = os.path.join(CACHE_DIR, "historical_calibration.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                blob = json.load(f)
            cal_data = blob.get("calibration", {})
            lw = cal_data.get("linear_weights")
            if lw:
                return {
                    "Out": 0.0,
                    "FieldersChoice": 0.0,
                    "Sacrifice": 0.0,
                    "Single": lw.get("single_w", 0.47),
                    "Double": lw.get("double_w", 0.78),
                    "Triple": lw.get("triple_w", 1.05),
                    "HomeRun": lw.get("hr_w", 1.40),
                    "Error": lw.get("single_w", 0.47),
                }
    except Exception:
        pass
    return {
        "Out": 0.0, "FieldersChoice": 0.0, "Sacrifice": 0.0,
        "Single": 0.47, "Double": 0.78, "Triple": 1.05, "HomeRun": 1.40,
        "Error": 0.47,
    }


def _fallback_contact_rv() -> Dict[str, float]:
    """Hardcoded fallback contact run values by pitch type."""
    return {
        "Fastball": 0.14,
        "Sinker": 0.12,
        "Cutter": 0.13,
        "Slider": 0.11,
        "Sweeper": 0.11,
        "Curveball": 0.10,
        "Knuckle Curve": 0.10,
        "Changeup": 0.12,
        "Splitter": 0.11,
    }


# ── 1D. ΔRE Computation ──────────────────────────────────────────────────────

def _get_re(re24: Dict[str, float], on1b: int, on2b: int, on3b: int, outs: int) -> float:
    """Lookup RE24 value, returning 0.0 for 3+ outs."""
    if outs >= 3:
        return 0.0
    return re24.get(_bo_key(on1b, on2b, on3b, outs), 0.0)


def compute_delta_re(
    pitch_type: str,
    count: str,
    base_out_state: Tuple[int, int, int, int],  # (on1b, on2b, on3b, outs)
    outcome_probs: PitchOutcomeProbs,
    re24: Dict[str, float],
    contact_rv: Dict[str, float],
    linear_weights: Dict[str, float],
) -> float:
    """Compute ΔRE for a pitch at the given game state.

    Lower ΔRE = better for pitcher (suppresses more runs).

    For each outcome:
    - Ball: count transition or walk
    - Called/swinging strike: count transition or strikeout
    - Foul: count transition (if s<2) or no change (if s=2)
    - In play: weighted average across BIP outcomes
    - HBP: same as walk
    """
    on1b, on2b, on3b, outs = base_out_state
    b, s = int(count[0]), int(count[2])
    re_current = _get_re(re24, on1b, on2b, on3b, outs)

    delta_re = 0.0

    # ── Ball ──
    if b < 3:
        # Count advances: next count RE (from count_rv perspective, the "state"
        # is just the count; base-out doesn't change on a ball)
        # We approximate: RE stays the same base-out state but count changes.
        # The count-level cost is captured in the count_rv transition.
        # For ΔRE we just keep current base-out state (ball doesn't change it).
        re_next = re_current  # base-out unchanged on ball
        delta_ball = re_next - re_current  # = 0 for base-out, but count worsens
        # The count worsening is implicit: the next pitch will be at a worse count.
        # To capture count-level effects, we use a simple approximation:
        # Getting deeper in the count raises RE proportionally.
        # Use a fixed ball cost per count from general data.
        ball_cost = _ball_cost_approx(b, s)
        delta_ball = ball_cost
    else:
        # Walk: forced advances
        n1, n2, n3, runs = _advance_runners_walk(on1b, on2b, on3b)
        re_after = _get_re(re24, n1, n2, n3, outs) + runs
        delta_ball = re_after - re_current

    delta_re += outcome_probs.p_ball * delta_ball

    # ── Called strike ──
    if s < 2:
        strike_gain = _strike_gain_approx(b, s)
        delta_cs = -strike_gain  # negative = good for pitcher
    else:
        # Strikeout
        re_after = _get_re(re24, on1b, on2b, on3b, outs + 1)
        delta_cs = re_after - re_current

    delta_re += outcome_probs.p_called_strike * delta_cs

    # ── Swinging strike ──
    if s < 2:
        strike_gain = _strike_gain_approx(b, s)
        delta_ss = -strike_gain
    else:
        re_after = _get_re(re24, on1b, on2b, on3b, outs + 1)
        delta_ss = re_after - re_current

    delta_re += outcome_probs.p_swinging_strike * delta_ss

    # ── Foul ──
    if s < 2:
        strike_gain = _strike_gain_approx(b, s)
        delta_foul = -strike_gain
    else:
        delta_foul = 0.0  # no count change on foul with 2 strikes

    delta_re += outcome_probs.p_foul * delta_foul

    # ── In play ──
    # Weighted average across BIP outcomes (out/1B/2B/3B/HR)
    crv = contact_rv.get(pitch_type, 0.12)
    lw = linear_weights

    # BIP outcome distribution from contact_rv:
    # We derive approximate outcome probabilities from the contact_rv value
    # using the linear weights as anchors.
    # Simpler approach: directly compute RE changes for each BIP outcome
    # and weight by approximate rates.

    # Approximate BIP outcome rates (league-average, can be refined)
    p_out_bip = 0.70  # ~70% of BIPs are outs
    p_single_bip = 0.17
    p_double_bip = 0.05
    p_triple_bip = 0.005
    p_hr_bip = 0.03
    p_error_bip = 0.025
    p_sac_bip = 0.02

    # Scale BIP outcome rates so their weighted wOBA ≈ contact_rv
    # First compute what the default rates imply
    default_crv = (p_single_bip * lw.get("single_w", 0.47) +
                   p_double_bip * lw.get("double_w", 0.78) +
                   p_triple_bip * lw.get("triple_w", 1.05) +
                   p_hr_bip * lw.get("hr_w", 1.40) +
                   p_error_bip * lw.get("single_w", 0.47))
    # Scale hit rates to match actual contact_rv
    if default_crv > 0:
        hit_scale = crv / default_crv
    else:
        hit_scale = 1.0
    hit_scale = max(0.3, min(3.0, hit_scale))  # clamp

    # Adjusted rates
    adj_single = p_single_bip * hit_scale
    adj_double = p_double_bip * hit_scale
    adj_triple = p_triple_bip * hit_scale
    adj_hr = p_hr_bip * hit_scale
    adj_error = p_error_bip * hit_scale
    adj_out = 1.0 - adj_single - adj_double - adj_triple - adj_hr - adj_error - p_sac_bip
    adj_out = max(0.3, adj_out)

    # Renormalize
    total_bip = adj_out + adj_single + adj_double + adj_triple + adj_hr + adj_error + p_sac_bip
    adj_out /= total_bip
    adj_single /= total_bip
    adj_double /= total_bip
    adj_triple /= total_bip
    adj_hr /= total_bip
    adj_error /= total_bip
    adj_sac = p_sac_bip / total_bip

    # Out on BIP
    re_out = _get_re(re24, on1b, on2b, on3b, outs + 1)
    delta_ip = adj_out * (re_out - re_current)

    # Sacrifice (out + R3 scores if R3 and < 2 outs)
    if on3b and outs < 2:
        re_sac = _get_re(re24, on1b, on2b, 0, outs + 1) + 1.0
    else:
        re_sac = _get_re(re24, on1b, on2b, on3b, outs + 1)
    delta_ip += adj_sac * (re_sac - re_current)

    # Single
    n1, n2, n3, runs = _advance_runners_single(on1b, on2b, on3b)
    re_single = _get_re(re24, n1, n2, n3, outs) + runs
    delta_ip += adj_single * (re_single - re_current)

    # Error (treat as single)
    delta_ip += adj_error * (re_single - re_current)

    # Double
    n1, n2, n3, runs = _advance_runners_double(on1b, on2b, on3b)
    re_double = _get_re(re24, n1, n2, n3, outs) + runs
    delta_ip += adj_double * (re_double - re_current)

    # Triple
    n1, n2, n3, runs = _advance_runners_triple(on1b, on2b, on3b)
    re_triple = _get_re(re24, n1, n2, n3, outs) + runs
    delta_ip += adj_triple * (re_triple - re_current)

    # HR
    n1, n2, n3, runs = _advance_runners_hr(on1b, on2b, on3b)
    re_hr = _get_re(re24, n1, n2, n3, outs) + runs
    delta_ip += adj_hr * (re_hr - re_current)

    delta_re += outcome_probs.p_in_play * delta_ip

    # ── HBP (same as walk) ──
    if outcome_probs.p_hbp > 0:
        n1, n2, n3, runs = _advance_runners_walk(on1b, on2b, on3b)
        re_hbp = _get_re(re24, n1, n2, n3, outs) + runs
        delta_re += outcome_probs.p_hbp * (re_hbp - re_current)

    return float(delta_re)


def _ball_cost_approx(b: int, s: int) -> float:
    """Approximate ball cost (RE increase) at a given count.

    These represent the average increase in run expectancy when a ball is added.
    Values from count_calibration analysis.
    """
    costs = {
        (0, 0): 0.030, (0, 1): 0.025, (0, 2): 0.020,
        (1, 0): 0.035, (1, 1): 0.030, (1, 2): 0.025,
        (2, 0): 0.045, (2, 1): 0.040, (2, 2): 0.035,
    }
    return costs.get((b, s), 0.030)


def _strike_gain_approx(b: int, s: int) -> float:
    """Approximate strike gain (RE decrease) at a given count.

    These represent the average decrease in run expectancy when a strike is added.
    """
    gains = {
        (0, 0): 0.035, (0, 1): 0.045,
        (1, 0): 0.035, (1, 1): 0.050,
        (2, 0): 0.040, (2, 1): 0.060,
        (3, 0): 0.045, (3, 1): 0.070,
    }
    return gains.get((b, s), 0.040)


# ── 1E. Caching ──────────────────────────────────────────────────────────────

def _parquet_fingerprint(path: str) -> Dict:
    try:
        return {"path": path, "mtime": os.path.getmtime(path), "size": os.path.getsize(path)}
    except OSError:
        return {"path": path, "mtime": None, "size": None}


def _cache_path() -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, "run_expectancy.json")


def _serialize_calibration(cal: RunExpectancyCalibration) -> dict:
    # Convert PitchOutcomeProbs in outcome_probs to dicts
    outcome_probs_ser = {}
    for pt, counts in cal.outcome_probs.items():
        outcome_probs_ser[pt] = {}
        for count_key, probs in counts.items():
            if isinstance(probs, PitchOutcomeProbs):
                outcome_probs_ser[pt][count_key] = probs.as_dict()
            else:
                outcome_probs_ser[pt][count_key] = probs

    return {
        "re24": cal.re24,
        "re24_n": cal.re24_n,
        "outcome_probs": outcome_probs_ser,
        "contact_rv": cal.contact_rv,
        "count_rv": cal.count_rv,
    }


def _deserialize_calibration(blob: dict) -> RunExpectancyCalibration:
    outcome_probs = {}
    for pt, counts in blob.get("outcome_probs", {}).items():
        outcome_probs[pt] = {}
        for count_key, probs_dict in counts.items():
            outcome_probs[pt][count_key] = PitchOutcomeProbs(**probs_dict)

    return RunExpectancyCalibration(
        re24=blob.get("re24", {}),
        re24_n={k: int(v) for k, v in blob.get("re24_n", {}).items()},
        outcome_probs=outcome_probs,
        contact_rv=blob.get("contact_rv", {}),
        count_rv=blob.get("count_rv", {}),
    )


def compute_run_expectancy_calibration(
    parquet_path: str = PARQUET_PATH,
) -> RunExpectancyCalibration:
    """Compute full RE calibration from Trackman parquet."""
    re24_matrix = compute_re24_matrix(parquet_path)
    outcome_probs = compute_pitch_outcome_probs(parquet_path)
    contact_rv = compute_contact_rv(parquet_path)

    # Load count RVs from existing count calibration cache
    count_rv = _load_count_rv()

    return RunExpectancyCalibration(
        re24=re24_matrix.matrix,
        re24_n=re24_matrix.n_obs,
        outcome_probs=outcome_probs,
        contact_rv=contact_rv,
        count_rv=count_rv,
    )


def _load_count_rv() -> Dict[str, float]:
    """Load count run values from count_calibration cache."""
    try:
        cache_file = os.path.join(CACHE_DIR, "count_calibration.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                blob = json.load(f)
            cal_data = blob.get("calibration", {})
            rv = cal_data.get("run_values")
            if rv:
                return rv
    except Exception:
        pass
    # Fallback count RVs
    return {
        "0-0": 0.28, "0-1": 0.21, "0-2": 0.10,
        "1-0": 0.33, "1-1": 0.27, "1-2": 0.15,
        "2-0": 0.39, "2-1": 0.34, "2-2": 0.21,
        "3-0": 0.43, "3-1": 0.40, "3-2": 0.28,
    }


def load_run_expectancy_calibration(
    parquet_path: str = PARQUET_PATH,
    force_refresh: bool = False,
) -> Optional[RunExpectancyCalibration]:
    """Load RE calibration from cache, recomputing if parquet changed.

    Returns None if no parquet file exists (graceful degradation).
    """
    if not os.path.exists(parquet_path):
        # Return fallback-based calibration
        return RunExpectancyCalibration(
            re24=_fallback_re24().matrix,
            re24_n=_fallback_re24().n_obs,
            outcome_probs={pt: {k: probs for k, probs in counts.items()}
                          for pt, counts in _fallback_outcome_probs_table().items()},
            contact_rv=_fallback_contact_rv(),
            count_rv=_load_count_rv(),
        )

    cache = _cache_path()
    fp = _parquet_fingerprint(parquet_path)

    if not force_refresh and os.path.exists(cache):
        try:
            with open(cache, "r", encoding="utf-8") as f:
                blob = json.load(f)
            if blob.get("fingerprint") == fp and "calibration" in blob:
                return _deserialize_calibration(blob["calibration"])
        except Exception:
            pass

    cal = compute_run_expectancy_calibration(parquet_path=parquet_path)
    try:
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(
                {"fingerprint": fp, "calibration": _serialize_calibration(cal)},
                f,
                indent=2,
            )
    except Exception:
        pass
    return cal


# ── Convenience: get RE24 value for a GameState ──────────────────────────────

def re24_lookup(re24: Dict[str, float], on1b: bool, on2b: bool, on3b: bool, outs: int) -> float:
    """Look up RE24 value for a game state."""
    return _get_re(re24, int(on1b), int(on2b), int(on3b), outs)
