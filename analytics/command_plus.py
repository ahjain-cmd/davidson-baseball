"""Command+ and pitch pair results computation.

Command+ measures pitch location quality using:
  1. Outcome-weighted location spread — deviations from the pitcher's mean
     location are weighted by pitch outcome.  Good outcomes (called strikes,
     whiffs) reduce the effective miss; bad outcomes (hard contact, balls)
     amplify it.
  2. Bayesian stabilization — small-sample Command+ is regressed toward
     league-average (100) so that 15-pitch cameos don't produce wild scores.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore

from config import (
    in_zone_mask, SWING_CALLS, filter_minor_pitches,
    MIN_TUNNEL_SEQ_PCT, PLATE_SIDE_MAX, PLATE_HEIGHT_MIN, PLATE_HEIGHT_MAX,
)
from data.loader import query_population

# ---------------------------------------------------------------------------
# Outcome-weight mapping: how much each pitch outcome inflates or
# deflates the location-miss penalty.
#   < 1.0  → good outcome, reduces miss penalty
#   = 1.0  → neutral
#   > 1.0  → bad outcome, amplifies miss penalty
# ---------------------------------------------------------------------------
_OUTCOME_WEIGHTS = {
    "StrikeCalled": 0.5,
    "StrikeSwinging": 0.5,
    "FoulBall": 0.75,
    "FoulBallNotFieldable": 0.75,
    "FoulBallFieldable": 0.75,
    "BallCalled": 1.2,
    "BallinDirt": 1.2,
    "BallIntentional": 1.0,
    "HitByPitch": 1.5,
    "InPlay": 1.0,  # base — overridden by EV tiers below
}

# ---------------------------------------------------------------------------
# Count-context multipliers: adjust outcome weights by the count.
# A ball on 0-0 is poor command; a ball on 0-2 is a strategic waste pitch.
# Keys are (balls, strikes) → {"ball": multiplier, "strike": multiplier}.
#   ball mult > 1  → throwing a ball here is WORSE than usual
#   ball mult < 1  → throwing a ball here is ACCEPTABLE (waste pitch)
#   strike mult < 1 → getting a strike here is MORE valuable than usual
# ---------------------------------------------------------------------------
_COUNT_CONTEXT = {
    # Pitcher's counts — waste pitches are strategic
    (0, 2): {"ball": 0.5, "strike": 0.9},
    (1, 2): {"ball": 0.6, "strike": 0.9},
    # Neutral / slight advantage
    (0, 0): {"ball": 1.1, "strike": 1.0},   # first pitch — get ahead
    (0, 1): {"ball": 0.9, "strike": 1.0},
    (1, 1): {"ball": 1.0, "strike": 1.0},
    (2, 2): {"ball": 0.9, "strike": 0.9},
    # Behind in count — balls are costly, strikes are crucial
    (1, 0): {"ball": 1.2, "strike": 0.85},
    (2, 0): {"ball": 1.4, "strike": 0.75},
    (2, 1): {"ball": 1.2, "strike": 0.85},
    (3, 0): {"ball": 1.5, "strike": 0.7},
    (3, 1): {"ball": 1.4, "strike": 0.75},
    # Full count — walk vs strikeout, high stakes both ways
    (3, 2): {"ball": 1.3, "strike": 0.8},
}

# Bayesian stabilization: number of pitches at which the observed score
# receives 50 % weight (the other 50 % comes from the league-average prior).
BAYESIAN_N = 150

# Pitch-call categories for count multiplier lookup
_BALL_CALLS = {"BallCalled", "BallinDirt", "BallIntentional", "HitByPitch"}
_STRIKE_CALLS = {
    "StrikeCalled", "StrikeSwinging",
    "FoulBall", "FoulBallNotFieldable", "FoulBallFieldable",
    "InPlay",
}


def _outcome_weight_series(ptd):
    """Per-pitch outcome weights combining base outcome + count context + EV."""
    w = ptd["PitchCall"].map(_OUTCOME_WEIGHTS).fillna(1.0).copy()

    # EV-based overrides for batted balls
    if "ExitSpeed" in ptd.columns:
        inplay = ptd["PitchCall"] == "InPlay"
        ev = pd.to_numeric(ptd["ExitSpeed"], errors="coerce")
        w.loc[inplay & (ev >= 98)]              = 2.0   # barrel-level
        w.loc[inplay & (ev >= 95) & (ev < 98)]  = 1.5   # hard hit
        w.loc[inplay & (ev < 85)]               = 0.8   # weak contact

    # Count-context multipliers
    if "Balls" in ptd.columns and "Strikes" in ptd.columns:
        balls = pd.to_numeric(ptd["Balls"], errors="coerce")
        strikes = pd.to_numeric(ptd["Strikes"], errors="coerce")
        for (b, s), mults in _COUNT_CONTEXT.items():
            count_mask = (balls == b) & (strikes == s)
            if not count_mask.any():
                continue
            ball_mask = count_mask & ptd["PitchCall"].isin(_BALL_CALLS)
            strike_mask = count_mask & ptd["PitchCall"].isin(_STRIKE_CALLS)
            if ball_mask.any():
                w.loc[ball_mask] *= mults["ball"]
            if strike_mask.any():
                w.loc[strike_mask] *= mults["strike"]

    return w


def _weighted_spread(ptd, weights):
    """Outcome-weighted location spread: sqrt( sum(w·d²) / sum(w) )."""
    h = ptd["PlateLocHeight"].values.astype(float)
    s = ptd["PlateLocSide"].values.astype(float)
    w = weights.values.astype(float)
    wsum = w.sum()
    if wsum == 0:
        return np.nan
    mean_h = np.average(h, weights=w)
    mean_s = np.average(s, weights=w)
    dev_sq = (h - mean_h) ** 2 + (s - mean_s) ** 2
    return np.sqrt(np.average(dev_sq, weights=w))


def _build_fallback_spreads(data, pitch_types):
    """Compute per-pitcher loc spreads from in-memory Davidson data as fallback."""
    if data is None or data.empty:
        return {}
    req = {"TaggedPitchType", "Pitcher", "PlateLocSide", "PlateLocHeight"}
    if not req.issubset(data.columns):
        return {}
    loc = data.dropna(subset=["PlateLocSide", "PlateLocHeight", "TaggedPitchType"]).copy()
    loc = loc[
        loc["PlateLocSide"].between(-PLATE_SIDE_MAX, PLATE_SIDE_MAX)
        & loc["PlateLocHeight"].between(PLATE_HEIGHT_MIN, PLATE_HEIGHT_MAX)
        & loc["TaggedPitchType"].isin(pitch_types)
    ]
    if loc.empty:
        return {}
    result = {}
    for pt, grp in loc.groupby("TaggedPitchType"):
        pitcher_spreads = []
        for _, pg in grp.groupby("Pitcher"):
            if len(pg) < 10:
                continue
            s = np.sqrt(pg["PlateLocHeight"].std()**2 + pg["PlateLocSide"].std()**2)
            if np.isfinite(s):
                pitcher_spreads.append(s)
        if len(pitcher_spreads) >= 3:
            result[pt] = np.array(pitcher_spreads)
    return result


def _compute_command_plus(pdf, data=None):
    """Compute Command+ for each pitch type. Returns DataFrame with
    Pitch, Pitches, Loc Spread (ft), Zone%, Edge%, CSW%, Chase%, Command+, Percentile.
    Returns empty DataFrame if insufficient data."""
    cmd_rows = []
    for pt in sorted(pdf["TaggedPitchType"].dropna().unique()):
        ptd = pdf[pdf["TaggedPitchType"] == pt].dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
        # Filter to valid location bounds to avoid outlier bias
        ptd = ptd[
            ptd["PlateLocSide"].between(-PLATE_SIDE_MAX, PLATE_SIDE_MAX)
            & ptd["PlateLocHeight"].between(PLATE_HEIGHT_MIN, PLATE_HEIGHT_MAX)
        ]
        if len(ptd) < 10:
            continue

        # Outcome-weighted location spread
        weights = _outcome_weight_series(ptd)
        loc_spread = _weighted_spread(ptd, weights)

        in_zone = in_zone_mask(ptd)
        zone_pct = in_zone.mean() * 100
        edge = (
            ((ptd["PlateLocSide"].abs().between(0.5, 1.1)) |
             (ptd["PlateLocHeight"].between(1.2, 1.8)) |
             (ptd["PlateLocHeight"].between(3.2, 3.8))) &
            (ptd["PlateLocSide"].abs() <= 1.5) &
            ptd["PlateLocHeight"].between(0.5, 4.5)
        )
        edge_pct = edge.mean() * 100
        csw = ptd["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"]).mean() * 100
        out_zone = ~in_zone
        chase_swings = ptd[out_zone & ptd["PitchCall"].isin(SWING_CALLS)]
        chase_pct = len(chase_swings) / max(out_zone.sum(), 1) * 100
        cmd_rows.append({
            "Pitch": pt,
            "Pitches": len(ptd),
            "Loc Spread (ft)": round(loc_spread, 2) if np.isfinite(loc_spread) else np.nan,
            "Zone%": round(zone_pct, 1),
            "Edge%": round(edge_pct, 1),
            "CSW%": round(csw, 1),
            "Chase%": round(chase_pct, 1),
        })
    if not cmd_rows:
        return pd.DataFrame()
    cmd_df = pd.DataFrame(cmd_rows)
    # NCAA D1 baseline via population table (DuckDB)
    # Population uses unweighted spreads — our pitcher's outcome-weighted
    # spread is compared against this, so pitchers who "get away with" misses
    # (called strikes on borderline pitches) are rewarded.
    pitch_types = sorted(cmd_df["Pitch"].unique())
    if pitch_types:
        pt_sql = ", ".join(f"'{p.replace(chr(39), chr(39)+chr(39))}'" for p in pitch_types)
        sql = f"""
            WITH t AS (
                SELECT Pitcher,
                       PlateLocSide,
                       PlateLocHeight,
                       CASE TaggedPitchType
                           WHEN 'FourSeamFastBall' THEN 'Fastball'
                           WHEN 'OneSeamFastBall' THEN 'Sinker'
                           WHEN 'TwoSeamFastBall' THEN 'Sinker'
                           WHEN 'ChangeUp' THEN 'Changeup'
                           ELSE TaggedPitchType
                       END AS pt_norm
                FROM trackman
                WHERE TaggedPitchType NOT IN ('Other','Undefined','Knuckleball')
                  AND PlateLocSide BETWEEN -{PLATE_SIDE_MAX} AND {PLATE_SIDE_MAX}
                  AND PlateLocHeight BETWEEN {PLATE_HEIGHT_MIN} AND {PLATE_HEIGHT_MAX}
            )
            SELECT pt_norm AS PitchType,
                   Pitcher,
                   STDDEV(PlateLocHeight) AS std_h,
                   STDDEV(PlateLocSide) AS std_s,
                   COUNT(*) AS n
            FROM t
            WHERE pt_norm IN ({pt_sql})
            GROUP BY pt_norm, Pitcher
            HAVING COUNT(*) >= 10
        """
        baseline_df = query_population(sql)
    else:
        baseline_df = pd.DataFrame()

    # Fallback: if D1 population query failed, use in-memory data
    fallback_spreads = {}
    if baseline_df.empty and data is not None:
        fallback_spreads = _build_fallback_spreads(data, pitch_types)

    cmd_scores = []
    cmd_pctls = []
    for _, row in cmd_df.iterrows():
        pt = row["Pitch"]
        n_pitches = row["Pitches"]
        my_spread = row["Loc Spread (ft)"]
        if pd.isna(my_spread):
            cmd_scores.append(100.0)
            cmd_pctls.append(50.0)
            continue

        # Try D1 population baseline first
        spreads = None
        if not baseline_df.empty:
            pt_df = baseline_df[baseline_df["PitchType"] == pt].copy()
            if not pt_df.empty:
                spreads = np.sqrt(pt_df["std_h"].astype(float)**2 + pt_df["std_s"].astype(float)**2)
                spreads = spreads.replace([np.inf, -np.inf], np.nan).dropna()
                if len(spreads) < 3:
                    spreads = None

        # Fallback to in-memory data
        if spreads is None and pt in fallback_spreads:
            spreads = fallback_spreads[pt]

        if spreads is None:
            cmd_scores.append(100.0)
            cmd_pctls.append(50.0)
            continue

        raw_pctl = 100 - percentileofscore(spreads, my_spread, kind="rank")

        # Bayesian stabilization: regress toward league-average (50th pctl)
        alpha = n_pitches / (n_pitches + BAYESIAN_N)
        stab_pctl = alpha * raw_pctl + (1 - alpha) * 50.0
        stab_cmd = 100 + (stab_pctl - 50) * 0.4

        cmd_scores.append(round(stab_cmd, 0))
        cmd_pctls.append(round(stab_pctl, 1))

    cmd_df["Command+"] = cmd_scores
    cmd_df["Percentile"] = cmd_pctls
    cmd_df = cmd_df.sort_values("Command+", ascending=False)
    return cmd_df


def _compute_pitch_pair_results(pdf, data, tunnel_df=None):
    """Compute effectiveness when pitch B follows pitch A in an at-bat."""
    if pdf.empty:
        return pd.DataFrame()
    pdf = filter_minor_pitches(pdf, min_pct=MIN_TUNNEL_SEQ_PCT)
    if pdf.empty:
        return pd.DataFrame()

    # Sort by game, at-bat, pitch number
    sort_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning", "PitchNo"] if c in pdf.columns]
    if len(sort_cols) < 2:
        return pd.DataFrame()

    pdf_s = pdf.sort_values(sort_cols).copy()
    pdf_s["PrevPitch"] = pdf_s.groupby(["GameID", "Batter", "PAofInning"])["TaggedPitchType"].shift(1)
    pdf_s = pdf_s.dropna(subset=["PrevPitch"])

    is_whiff = pdf_s["PitchCall"] == "StrikeSwinging"
    is_swing = pdf_s["PitchCall"].isin(SWING_CALLS)
    is_csw = pdf_s["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"])

    _TB_MAP = {"Single": 1, "Double": 2, "Triple": 3, "HomeRun": 4}

    rows = []
    for (prev, curr), grp in pdf_s.groupby(["PrevPitch", "TaggedPitchType"]):
        n = len(grp)
        if n < 25:
            continue
        swings = grp[is_swing.reindex(grp.index, fill_value=False)]
        whiffs = grp[is_whiff.reindex(grp.index, fill_value=False)]
        csws = grp[is_csw.reindex(grp.index, fill_value=False)]
        batted = grp[(grp["PitchCall"] == "InPlay") & grp["ExitSpeed"].notna()]
        # SLG on batted ball events
        inplay = grp[grp["PitchCall"] == "InPlay"]
        if len(inplay) > 0 and "PlayResult" in grp.columns:
            total_bases = inplay["PlayResult"].map(_TB_MAP).fillna(0).sum()
            slg = round(total_bases / len(inplay), 3)
        else:
            slg = np.nan
        # Putaway% proxy: 2-strike pitches resulting in swinging/called strike
        if "Strikes" in grp.columns:
            two_strike = grp[grp["Strikes"] == 2]
            putaway = two_strike[two_strike["PitchCall"].isin(["StrikeSwinging", "StrikeCalled"])]
            putaway_pct = len(putaway) / max(len(two_strike), 1) * 100 if len(two_strike) > 0 else np.nan
        else:
            putaway_pct = np.nan
        # K%: use KorBB if available, otherwise PlayResult == Strikeout
        if "KorBB" in grp.columns:
            k_events = grp["KorBB"].isin(["Strikeout", "K"])
            k_pct = k_events.mean() * 100 if len(grp) > 0 else np.nan
        elif "PlayResult" in grp.columns:
            k_events = grp["PlayResult"].isin(["Strikeout", "K"])
            k_pct = k_events.mean() * 100 if len(grp) > 0 else np.nan
        else:
            k_pct = np.nan
        # Tunnel lookup for this pair
        tun_grade, tun_score = "-", np.nan
        if isinstance(tunnel_df, pd.DataFrame) and not tunnel_df.empty:
            tun_match = tunnel_df[
                ((tunnel_df["Pitch A"] == prev) & (tunnel_df["Pitch B"] == curr)) |
                ((tunnel_df["Pitch A"] == curr) & (tunnel_df["Pitch B"] == prev))
            ]
            if not tun_match.empty:
                tun_grade = tun_match.iloc[0]["Grade"]
                tun_score = tun_match.iloc[0]["Tunnel Score"]
        rows.append({
            "Setup Pitch": prev, "Follow Pitch": curr, "Count": n,
            "Whiff%": round(len(whiffs) / max(len(swings), 1) * 100, 1),
            "CSW%": round(len(csws) / n * 100, 1),
            "Avg EV": round(batted["ExitSpeed"].mean(), 1) if len(batted) > 0 else np.nan,
            "SLG": slg,
            "Putaway%": round(putaway_pct, 1) if not pd.isna(putaway_pct) else np.nan,
            "K%": round(k_pct, 1) if not pd.isna(k_pct) else np.nan,
            "Chase%": round(
                (lambda _iz=in_zone_mask(grp): len(grp[(~_iz) & grp["PitchCall"].isin(SWING_CALLS)]) /
                max(len(grp[~_iz]), 1) * 100)(), 1),
            "Tunnel": tun_grade,
            "Tunnel Score": tun_score,
        })
    return pd.DataFrame(rows).sort_values("Whiff%", ascending=False).reset_index(drop=True)
