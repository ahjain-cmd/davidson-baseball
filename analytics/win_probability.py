"""
Win Probability (WP) Model — empirical WP table, WP-based leverage, and ΔWP.

Built from Trackman play-by-play data, paralleling the RE24 infrastructure in
``run_expectancy.py``.  Uses the same cache-with-fingerprint pattern.

State key: (inning_bucket, half, score_diff_bucket, base_out_state)
  - inning_bucket: "early" (1-3), "mid" (4-6), "late" (7-9), "extra" (10+)
  - half: "top" or "bot"
  - score_diff_bucket: clamped to [-5, +5], home perspective
  - base_out_state: "on1b,on2b,on3b,outs" (same as RE24)

WP is always from the **home team's** perspective (0.0–1.0).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import numpy as np

from config import CACHE_DIR, PARQUET_PATH


# ── Inning bucketing ─────────────────────────────────────────────────────────

def _inning_bucket(inning: int) -> str:
    if inning <= 3:
        return "early"
    if inning <= 6:
        return "mid"
    if inning <= 9:
        return "late"
    return "extra"


def _score_diff_bucket(diff: int) -> int:
    """Clamp score differential (home - away) to [-5, +5]."""
    return max(-5, min(5, diff))


def _wp_key(inning_bucket: str, half: str, score_diff: int,
            on1b: int, on2b: int, on3b: int, outs: int) -> str:
    sd = _score_diff_bucket(score_diff)
    return f"{inning_bucket}|{half}|{sd}|{on1b},{on2b},{on3b},{outs}"


# ── Runner-advance helpers (reuse from run_expectancy) ───────────────────────

def _advance_runners_single(on1b, on2b, on3b):
    return 1, on1b, on2b, on3b  # batter->1B, R1->2B, R2->3B, R3 scores

def _advance_runners_double(on1b, on2b, on3b):
    return 0, 1, on1b, on2b + on3b  # batter->2B, R1->3B, R2+R3 score

def _advance_runners_hr(on1b, on2b, on3b):
    return 0, 0, 0, 1 + on1b + on2b + on3b

def _advance_runners_walk(on1b, on2b, on3b):
    new1, new2, new3, runs = 1, on2b, on3b, 0
    if on1b:
        new2 = 1
        if on2b:
            new3 = 1
            if on3b:
                runs = 1
    return new1, new2, new3, runs


# ── WP table dataclass ──────────────────────────────────────────────────────

@dataclass
class WinProbabilityTable:
    """Empirical win probability lookup table."""
    wp: Dict[str, float]      # wp_key -> P(home wins)
    n_obs: Dict[str, int]     # wp_key -> observation count
    n_games: int              # total games used


# ── Build WP table from Trackman ─────────────────────────────────────────────

def build_wp_table(parquet_path: str = PARQUET_PATH) -> WinProbabilityTable:
    """Build empirical win probability table from Trackman play-by-play.

    For each game, reconstructs score + base-out state at each PA, determines
    the final winner, and aggregates P(home wins | state).
    """
    if not os.path.exists(parquet_path):
        return _fallback_wp_table()

    pq = parquet_path.replace("'", "''")
    con = duckdb.connect(database=":memory:")

    # Get PA-level data with enough info to track score and base-out state
    rows = con.execute(f"""
        WITH pa_data AS (
            SELECT
                GameID,
                CAST(Inning AS INTEGER) AS inning,
                BatterTeam,
                HomeTeam,
                AwayTeam,
                CAST(PAofInning AS INTEGER) AS pa_of_inning,
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
        SELECT GameID, inning, BatterTeam, HomeTeam, AwayTeam, pa_of_inning,
               outcome, runs_scored
        FROM pa_data
        WHERE outcome IS NOT NULL
        ORDER BY GameID, inning, pa_of_inning
    """).fetchall()
    con.close()

    # Group PAs by game
    games: Dict[str, List[Dict]] = {}
    game_teams: Dict[str, Tuple[str, str]] = {}  # GameID -> (HomeTeam, AwayTeam)
    for game_id, inning, batter_team, home_team, away_team, pa_of_inning, outcome, runs_scored in rows:
        if game_id not in games:
            games[game_id] = []
            game_teams[game_id] = (home_team, away_team)
        games[game_id].append({
            "inning": int(inning) if inning is not None else 1,
            "batter_team": batter_team,
            "pa_of_inning": pa_of_inning,
            "outcome": outcome,
            "runs_scored": int(runs_scored),
        })

    # Process each game: track score, base-out, determine winner
    state_wins: Dict[str, List[int]] = {}  # wp_key -> [1 if home won, 0 if away won]

    for game_id, pa_list in games.items():
        home_team, away_team = game_teams[game_id]
        if not pa_list or len(pa_list) < 10:
            continue  # skip incomplete games

        # Track running score
        home_score = 0
        away_score = 0

        # Determine final winner by total runs
        total_home_runs = sum(
            pa["runs_scored"] for pa in pa_list if pa["batter_team"] == home_team
        )
        total_away_runs = sum(
            pa["runs_scored"] for pa in pa_list if pa["batter_team"] == away_team
        )
        if total_home_runs == total_away_runs:
            continue  # skip ties (incomplete games)

        home_won = 1 if total_home_runs > total_away_runs else 0

        # Walk through PAs, tracking state at each PA
        on1b, on2b, on3b, outs = 0, 0, 0, 0
        prev_inning = 0
        prev_batter_team = None

        for pa in pa_list:
            inning = pa["inning"]
            bt = pa["batter_team"]

            # Detect half-inning change
            if inning != prev_inning or bt != prev_batter_team:
                on1b, on2b, on3b, outs = 0, 0, 0, 0
                prev_inning = inning
                prev_batter_team = bt

            if outs >= 3:
                continue

            # Record state BEFORE this PA
            is_top = (bt == away_team)
            half = "top" if is_top else "bot"
            score_diff = home_score - away_score  # home perspective
            ib = _inning_bucket(inning)
            key = _wp_key(ib, half, score_diff, on1b, on2b, on3b, outs)

            if key not in state_wins:
                state_wins[key] = []
            state_wins[key].append(home_won)

            # Update state based on outcome
            outcome = pa["outcome"]
            runs = pa["runs_scored"]

            if outcome in ("Strikeout", "Out"):
                outs += 1
            elif outcome == "Single" or outcome == "Error":
                n1, n2, n3, _ = _advance_runners_single(on1b, on2b, on3b)
                on1b, on2b, on3b = n1, n2, n3
            elif outcome == "Double":
                n1, n2, n3, _ = _advance_runners_double(on1b, on2b, on3b)
                on1b, on2b, on3b = n1, n2, n3
            elif outcome == "Triple":
                on1b, on2b, on3b = 0, 0, 1
            elif outcome == "HomeRun":
                on1b, on2b, on3b = 0, 0, 0
            elif outcome in ("Walk", "HitByPitch"):
                n1, n2, n3, _ = _advance_runners_walk(on1b, on2b, on3b)
                on1b, on2b, on3b = n1, n2, n3
            elif outcome == "Sacrifice":
                if on3b and outs < 2:
                    on3b = 0
                outs += 1
            elif outcome == "FieldersChoice":
                if on1b and outs < 2:
                    outs += 2
                    on1b = 0
                else:
                    outs += 1
            else:
                outs += 1

            # Credit runs to the batting team
            if is_top:
                away_score += runs
            else:
                home_score += runs

    # Compute empirical WP with Bayesian smoothing
    wp: Dict[str, float] = {}
    n_obs: Dict[str, int] = {}

    for key, wins_list in state_wins.items():
        n = len(wins_list)
        empirical = sum(wins_list) / n

        # Parse score_diff from key for logistic prior
        parts = key.split("|")
        sd = int(parts[2])
        prior_wp = 1.0 / (1.0 + math.exp(-0.7 * sd))

        # Bayesian shrinkage: weight empirical by sample size
        n_prior = 30
        w = n / (n + n_prior)
        smoothed = w * empirical + (1.0 - w) * prior_wp

        wp[key] = round(smoothed, 5)
        n_obs[key] = n

    return WinProbabilityTable(wp=wp, n_obs=n_obs, n_games=len(games))


# ── Fallback WP table (MLB-derived, D1-scaled) ──────────────────────────────

def _fallback_wp_table() -> WinProbabilityTable:
    """Hardcoded baseline WP table using logistic model.

    WP = 1 / (1 + exp(-k * score_diff - inning_adj))
    Calibrated to approximate MLB win probability curves.
    """
    wp = {}
    n_obs = {}

    for ib in ("early", "mid", "late", "extra"):
        # Inning weight: later innings lock in score advantages more
        inning_mult = {"early": 0.5, "mid": 0.7, "late": 1.0, "extra": 1.2}[ib]

        for half in ("top", "bot"):
            # Slight home advantage for bottom half
            home_adj = 0.02 if half == "bot" else 0.0

            for sd in range(-5, 6):
                for outs in range(3):
                    for b1 in range(2):
                        for b2 in range(2):
                            for b3 in range(2):
                                # Runners on base slightly favor batting team
                                runner_adj = (b1 + b2 + b3) * 0.01
                                if half == "top":
                                    runner_adj = -runner_adj  # away batting: hurts home
                                # Outs adjustment: more outs slightly reduce threat
                                outs_adj = -outs * 0.005 if half == "top" else outs * 0.005

                                logit = 0.7 * sd * inning_mult + home_adj + runner_adj + outs_adj
                                prob = 1.0 / (1.0 + math.exp(-logit))
                                key = _wp_key(ib, half, sd, b1, b2, b3, outs)
                                wp[key] = round(prob, 5)
                                n_obs[key] = 0

    return WinProbabilityTable(wp=wp, n_obs=n_obs, n_games=0)


# ── WP Lookup ────────────────────────────────────────────────────────────────

def lookup_wp(
    wp_table: WinProbabilityTable,
    inning: int,
    half: str,
    score_diff: int,
    on1b: int,
    on2b: int,
    on3b: int,
    outs: int,
) -> float:
    """Look up win probability for a game state. Returns home team WP."""
    ib = _inning_bucket(inning)
    key = _wp_key(ib, half, _score_diff_bucket(score_diff), on1b, on2b, on3b, outs)
    if key in wp_table.wp:
        return wp_table.wp[key]

    # Fallback: try without runners (more observations)
    key_no_runners = _wp_key(ib, half, _score_diff_bucket(score_diff), 0, 0, 0, outs)
    if key_no_runners in wp_table.wp:
        return wp_table.wp[key_no_runners]

    # Last resort: logistic prior based on score diff
    return 1.0 / (1.0 + math.exp(-0.7 * _score_diff_bucket(score_diff)))


# ── WP-Based Leverage ────────────────────────────────────────────────────────

def compute_wp_leverage(
    wp_table: WinProbabilityTable,
    inning: int,
    half: str,
    score_diff: int,
    on1b: int,
    on2b: int,
    on3b: int,
    outs: int,
) -> float:
    """Compute WP-based leverage index (0.0–1.0).

    Leverage = how much the current WP can swing on the next play.
    Measured as |WP_after_HR - WP_after_K|.
    """
    wp_now = lookup_wp(wp_table, inning, half, score_diff, on1b, on2b, on3b, outs)

    # Best case for batting team: HR (all runners + batter score)
    runs_on_hr = 1 + on1b + on2b + on3b
    if half == "top":
        # Away team batting: runs hurt home team
        sd_after_hr = score_diff - runs_on_hr
    else:
        # Home team batting: runs help home team
        sd_after_hr = score_diff + runs_on_hr
    wp_after_hr = lookup_wp(wp_table, inning, half, sd_after_hr, 0, 0, 0, outs)

    # Worst case for batting team: strikeout
    new_outs = outs + 1
    if new_outs >= 3:
        # Inning over: flip to other half or next inning
        if half == "top":
            wp_after_k = lookup_wp(wp_table, inning, "bot", score_diff, 0, 0, 0, 0)
        else:
            wp_after_k = lookup_wp(wp_table, inning + 1, "top", score_diff, 0, 0, 0, 0)
    else:
        wp_after_k = lookup_wp(wp_table, inning, half, score_diff, on1b, on2b, on3b, new_outs)

    leverage = abs(wp_after_hr - wp_after_k)
    # Normalize: typical WP swing ~0.03–0.15 → scale to [0, 1]
    return float(min(1.0, leverage / 0.15))


# ── ΔWP Computation ─────────────────────────────────────────────────────────

def compute_delta_wp(
    pitch_type: str,
    count: str,
    base_out_state: Tuple[int, int, int, int],  # (on1b, on2b, on3b, outs)
    outcome_probs: Any,  # PitchOutcomeProbs from run_expectancy
    wp_table: WinProbabilityTable,
    score_diff: int,
    inning: int,
    half: str,
    contact_rv: Dict[str, float],
    linear_weights: Dict[str, float],
    bip_profile: Optional[Dict[str, Dict[str, float]]] = None,
    gb_dp_rates: Optional[Dict[str, Any]] = None,
    pitcher_gb_pct: Optional[float] = None,
    pitcher_fb_pct: Optional[float] = None,
    tag_up_score_pct: Optional[float] = None,
    count_ball_cost: Optional[Dict[str, float]] = None,
    count_strike_gain: Optional[Dict[str, float]] = None,
) -> float:
    """Compute ΔWP for a pitch at the given game state.

    Parallels ``compute_delta_re()`` but uses win probability instead of run
    expectancy.  Lower ΔWP = better for pitcher (from pitcher's perspective,
    pitcher wants to reduce the batting team's chances).

    For home team batting: negative ΔWP means WP decreased = good for away pitcher.
    For away team batting: we flip perspective so negative always = good for pitcher.
    """
    on1b, on2b, on3b, outs = base_out_state
    b, s = int(count[0]), int(count[2])

    # Current WP (home perspective)
    wp_now = lookup_wp(wp_table, inning, half, score_diff, on1b, on2b, on3b, outs)

    # Pitcher perspective: if home team is batting, pitcher is away →
    # pitcher wants WP to decrease (home WP goes down).
    # If away team is batting, pitcher is home → pitcher wants WP to increase.
    # Sign convention: negative ΔWP = good for pitcher (same as ΔRE).
    batting_home = (half == "bot")
    sign = 1.0 if batting_home else -1.0  # +1 means WP increase hurts pitcher

    delta_wp = 0.0

    def _wp_after(sd, o1, o2, o3, o, inn=inning, h=half):
        """Look up WP after a state transition."""
        if o >= 3:
            # Inning over
            if h == "top":
                return lookup_wp(wp_table, inn, "bot", sd, 0, 0, 0, 0)
            else:
                return lookup_wp(wp_table, inn + 1, "top", sd, 0, 0, 0, 0)
        return lookup_wp(wp_table, inn, h, sd, o1, o2, o3, o)

    # ── Ball ──
    if b < 3:
        # Count transition: ball added. Use simple WP estimate for new count.
        # Approximate: ball moves WP slightly toward batting team.
        # We don't track count in WP table, so use a scaled estimate.
        ball_wp_delta = 0.002  # ~0.2% WP per ball (empirical average)
        delta_ball = sign * ball_wp_delta
    else:
        # Walk: forced advances
        n1, n2, n3, runs = _advance_runners_walk(on1b, on2b, on3b)
        new_sd = score_diff + (runs if batting_home else -runs)
        wp_after_walk = _wp_after(new_sd, n1, n2, n3, outs)
        delta_ball = (wp_after_walk - wp_now) * sign

    delta_wp += outcome_probs.p_ball * delta_ball

    # ── Called Strike ──
    if s < 2:
        strike_wp_delta = -0.002  # strike hurts batting team ~0.2% WP
        delta_cs = sign * strike_wp_delta
    else:
        # Strikeout
        wp_after_k = _wp_after(score_diff, on1b, on2b, on3b, outs + 1)
        delta_cs = (wp_after_k - wp_now) * sign

    delta_wp += outcome_probs.p_called_strike * delta_cs

    # ── Swinging Strike ──
    if s < 2:
        delta_ss = sign * (-0.003)  # swinging strike slightly worse for hitter
    else:
        wp_after_k = _wp_after(score_diff, on1b, on2b, on3b, outs + 1)
        delta_ss = (wp_after_k - wp_now) * sign

    delta_wp += outcome_probs.p_swinging_strike * delta_ss

    # ── Foul ──
    if s < 2:
        delta_foul = sign * (-0.001)
    else:
        delta_foul = 0.0

    delta_wp += outcome_probs.p_foul * delta_foul

    # ── In Play ──
    # Use BIP profile for outcome distribution
    crv = contact_rv.get(pitch_type, 0.12)
    lw = linear_weights

    if bip_profile and pitch_type in bip_profile:
        bp = bip_profile[pitch_type]
    else:
        bp = {"p_out": 0.70, "p_single": 0.17, "p_double": 0.05,
              "p_triple": 0.005, "p_hr": 0.03, "p_error": 0.025, "p_sac": 0.02}

    p_out = bp.get("p_out", 0.70)
    p_single = bp.get("p_single", 0.17)
    p_double = bp.get("p_double", 0.05)
    p_triple = bp.get("p_triple", 0.005)
    p_hr = bp.get("p_hr", 0.03)
    p_error = bp.get("p_error", 0.025)
    p_sac = bp.get("p_sac", 0.02)

    # Scale hit rates by contact quality (same as ΔRE)
    default_crv = (p_single * lw.get("Single", 0.47) +
                   p_double * lw.get("Double", 0.78) +
                   p_triple * lw.get("Triple", 1.05) +
                   p_hr * lw.get("HomeRun", 1.40) +
                   p_error * lw.get("Single", 0.47))
    hit_scale = max(0.3, min(3.0, crv / default_crv if default_crv > 0 else 1.0))

    adj_single = p_single * hit_scale
    adj_double = p_double * hit_scale
    adj_triple = p_triple * hit_scale
    adj_hr = p_hr * hit_scale
    adj_error = p_error * hit_scale
    adj_out = max(0.3, 1.0 - adj_single - adj_double - adj_triple - adj_hr - adj_error - p_sac)

    total_bip = adj_out + adj_single + adj_double + adj_triple + adj_hr + adj_error + p_sac
    adj_out /= total_bip
    adj_single /= total_bip
    adj_double /= total_bip
    adj_triple /= total_bip
    adj_hr /= total_bip
    adj_error /= total_bip
    adj_sac = p_sac / total_bip

    delta_ip = 0.0

    # Out on BIP (with DP modelling)
    _gb_pct = 0.43
    _dp_rate = 0.06
    if pitcher_gb_pct is not None:
        _gb_pct = pitcher_gb_pct / 100.0 if pitcher_gb_pct > 1.0 else pitcher_gb_pct
    if gb_dp_rates:
        _pt_gdp = gb_dp_rates.get(pitch_type, {})
        if _pt_gdp:
            if pitcher_gb_pct is None:
                _gb_pct = _pt_gdp.get("gb_pct", 43.0) / 100.0
            if _pt_gdp.get("gb_pct", 0) > 0:
                _dp_rate = _pt_gdp.get("dp_pct", 2.5) / _pt_gdp["gb_pct"]

    wp_after_out = _wp_after(score_diff, on1b, on2b, on3b, outs + 1)
    if on1b and outs < 2:
        p_dp = min(_gb_pct * _dp_rate, 0.15)
        wp_after_dp = _wp_after(score_diff, 0, on2b, on3b, outs + 2)
        delta_ip += adj_out * ((1.0 - p_dp) * ((wp_after_out - wp_now) * sign) +
                               p_dp * ((wp_after_dp - wp_now) * sign))
    else:
        delta_ip += adj_out * (wp_after_out - wp_now) * sign

    # Tag-up scoring
    if on3b and outs < 2:
        _fb_pct = pitcher_fb_pct / 100.0 if pitcher_fb_pct is not None and pitcher_fb_pct > 1.0 else (
            pitcher_fb_pct if pitcher_fb_pct is not None else None)
        air_pct = _fb_pct if _fb_pct is not None else (1.0 - _gb_pct)
        _tag_pct = tag_up_score_pct if tag_up_score_pct is not None else 0.40
        tag_rate = _tag_pct * air_pct
        tag_sd = score_diff + (1 if batting_home else -1)
        wp_tag = _wp_after(tag_sd, on1b, on2b, 0, outs + 1)
        delta_ip += adj_out * tag_rate * ((wp_tag - wp_now) * sign - (wp_after_out - wp_now) * sign)

    # Sacrifice
    if on3b and outs < 2:
        sac_sd = score_diff + (1 if batting_home else -1)
        wp_sac = _wp_after(sac_sd, on1b, on2b, 0, outs + 1)
    else:
        wp_sac = _wp_after(score_diff, on1b, on2b, on3b, outs + 1)
    delta_ip += adj_sac * (wp_sac - wp_now) * sign

    # Single
    n1, n2, n3, runs = _advance_runners_single(on1b, on2b, on3b)
    single_sd = score_diff + (runs if batting_home else -runs)
    wp_single = _wp_after(single_sd, n1, n2, n3, outs)
    delta_ip += adj_single * (wp_single - wp_now) * sign

    # Error (treat as single)
    delta_ip += adj_error * (wp_single - wp_now) * sign

    # Double
    n1, n2, n3, runs = _advance_runners_double(on1b, on2b, on3b)
    double_sd = score_diff + (runs if batting_home else -runs)
    wp_double = _wp_after(double_sd, n1, n2, n3, outs)
    delta_ip += adj_double * (wp_double - wp_now) * sign

    # Triple
    runs_triple = on1b + on2b + on3b
    triple_sd = score_diff + (runs_triple if batting_home else -runs_triple)
    wp_triple = _wp_after(triple_sd, 0, 0, 1, outs)
    delta_ip += adj_triple * (wp_triple - wp_now) * sign

    # HR
    runs_hr = 1 + on1b + on2b + on3b
    hr_sd = score_diff + (runs_hr if batting_home else -runs_hr)
    wp_hr = _wp_after(hr_sd, 0, 0, 0, outs)
    delta_ip += adj_hr * (wp_hr - wp_now) * sign

    delta_wp += outcome_probs.p_in_play * delta_ip

    # ── HBP (same as walk) ──
    if outcome_probs.p_hbp > 0:
        n1, n2, n3, runs = _advance_runners_walk(on1b, on2b, on3b)
        hbp_sd = score_diff + (runs if batting_home else -runs)
        wp_hbp = _wp_after(hbp_sd, n1, n2, n3, outs)
        delta_wp += outcome_probs.p_hbp * (wp_hbp - wp_now) * sign

    return float(delta_wp)


# ── Cache management ─────────────────────────────────────────────────────────

def _cache_path() -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, "win_probability.json")


def _parquet_fingerprint(parquet_path: str) -> str:
    try:
        st = os.stat(parquet_path)
        return f"{st.st_size}_{st.st_mtime_ns}"
    except Exception:
        return "unknown"


def load_wp_table(
    parquet_path: str = PARQUET_PATH,
    force_refresh: bool = False,
) -> WinProbabilityTable:
    """Load WP table from cache, recomputing if parquet changed."""
    if not os.path.exists(parquet_path):
        return _fallback_wp_table()

    cache = _cache_path()
    fp = _parquet_fingerprint(parquet_path)

    if not force_refresh and os.path.exists(cache):
        try:
            with open(cache, "r", encoding="utf-8") as f:
                blob = json.load(f)
            if blob.get("fingerprint") == fp and "wp" in blob:
                return WinProbabilityTable(
                    wp=blob["wp"],
                    n_obs={k: int(v) for k, v in blob.get("n_obs", {}).items()},
                    n_games=int(blob.get("n_games", 0)),
                )
        except Exception:
            pass

    table = build_wp_table(parquet_path=parquet_path)
    try:
        with open(cache, "w", encoding="utf-8") as f:
            json.dump({
                "fingerprint": fp,
                "wp": table.wp,
                "n_obs": table.n_obs,
                "n_games": table.n_games,
            }, f, indent=2)
    except Exception:
        pass

    return table
