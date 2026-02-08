# Davidson In-Game Decision Engine — Implementation Plan

## Executive Summary

This plan is grounded in a thorough audit of your actual codebase (`scouting.py`, `truemedia_api.py`, `loader.py`, `stats.py`, `sequencing.py`, `defense.py`, `catcher.py`) and sample data (140K-row Trackman parquet, raw 167-column CSVs, player positioning data, bat tracking JSON). The good news: you have far more infrastructure than most college programs attempting this. The realistic news: there are specific data gaps, architectural decisions, and sample-size constraints that must be addressed before any "engine" is trustworthy in-game.

---

## Part 1: What You Actually Have (Data Audit)

### 1.1 Trackman Parquet (Your Davidson Data)

**Schema:** 54 columns across 140K rows, 12,533 games, spanning seasons 0/2024/2025/2026.

**Fully populated fields (0% missing):**
- Game structure: `GameID`, `Date`, `PitchNo`, `Inning`, `PAofInning`, `PitchofPA`, `Outs`, `Balls`, `Strikes`
- Identity: `Pitcher`, `PitcherTeam`, `Batter`, `BatterTeam`, `BatterSide`, `PitcherThrows` (34 nulls only)
- Pitch classification: `TaggedPitchType`, `PitchCall`, `PlayResult`, `RunsScored`, `OutsOnPlay`
- Pitch physics: `RelSpeed`, `SpinRate`, `SpinAxis`, `InducedVertBreak`, `HorzBreak`, `RelHeight`, `RelSide`, `Extension`, `VertApprAngle`
- Plate location: `PlateLocSide`, `PlateLocHeight` (1,363 nulls = 0.97% — excellent)

**Partially populated (high null rate, expected):**
- Batted ball: `ExitSpeed` (103K nulls — only populated on InPlay events, correct behavior), `Angle`, `Direction`, `Distance`
- Trajectory: `vx0/vy0/vz0`, `x0/y0/z0`, `ax0/ay0/az0` (raw pitch trajectory, rarely used directly)

**Critical missing columns vs raw CSV:**
Your raw Trackman CSV has 167 columns. The parquet drops 113 of them. Key losses:

| Missing from Parquet | Impact on Decision Engine |
|---|---|
| `EffectiveVelo` | Used in swing timing analysis, contact depth calc, EffV-gap sequencing |
| `AutoPitchType` | Needed for pitch-type reconciliation (Trackman auto vs tagged) |
| `Catcher`, `CatcherId`, `CatcherTeam` | Already used in `catcher.py` but only from Davidson data load |
| `Top/Bottom` | Needed to determine home/away, reconstruct base state |
| `PopTime`, `ExchangeTime` | Catcher analytics, baserunning decisions |
| `PositionAt110X/Y/Z` | Landing position for spray chart accuracy |
| `HangTime` | Outfield positioning decisions |
| `ContactPositionX/Y/Z` | Contact point analysis |
| `ZoneSpeed`, `ZoneTime` | Pitch arrival time — useful for sequencing timing |
| `PitchUID`, `PlayID` | Needed to join positioning data to pitch data |
| Player positioning CSV | 7 fielder X/Z coordinates at release, `DetectedShift`, `FHC` |
| Bat tracking JSON | Currently empty (`Plays: []`) — future potential |
| All confidence columns | `PitchReleaseConfidence`, etc. — for data quality filtering |

**Recommendation:** Re-build the parquet pipeline to include at minimum: `EffectiveVelo`, `AutoPitchType`, `Top/Bottom`, `Catcher`, `CatcherTeam`, `PopTime`, `ExchangeTime`, `PitchUID`, `PlayID`, `HangTime`, `ZoneSpeed`, and all confidence columns. This is a one-time ETL change with massive downstream value.

### 1.2 TrueMedia API (Opposition Data)

**Endpoints confirmed working in your code:**

| Endpoint | What It Returns | How You Use It |
|---|---|---|
| `AllTeams` | Team list + IDs per season | Team selector |
| `TeamGames` | All games for a team | Feed into pitch-level fetch |
| `GamePitchesTrackman` | Full pitch-by-pitch with Trackman data | Opposition pitch-level analysis |
| `PlayerTotals` | Aggregate stats (rate/counting/discipline/exit) | Scouting reports, percentiles |
| `PlayerTotals` + `filters` | Count splits (ahead/behind/0-0/2-strike/etc.) | Count-specific game plans |
| `PlayerTotals` + pitcher hand filter | Platoon splits (vs LHP / vs RHP) | Platoon advantage scoring |
| `TeamTotals` | Team-level aggregates | D1 rankings context |

**Column coverage confirmed:**
- Hitting: PA, AB, AVG, OBP, SLG, OPS, wOBA, K%, BB%, ISO, ExitVel, Barrel%, HardHit%, Chase%, SwStrk%, Contact%, Swing%, Ground%/Fly%/Line%/Popup%, Pull%/Ctr%/Oppo%, pitch type usage, zone location percentages
- Pitching: IP, ERA, FIP, xFIP, WHIP, K/9, BB/9, FBVel, Spin, IVB, HB, same discipline/exit/batted-ball columns

**Coordinate conversion (the 0.83 factor):**
Your code at `truemedia_api.py:864` does:
```python
PlateLocSide = -pxNorm * 0.83
PlateLocHeight = pzNorm * 1.0 + 2.5
```
The `pxNorm` documentation says ±1 = plate edge. The plate is 17 inches = 1.417 feet wide, so half-plate = 0.708 feet. The 0.83 factor implies TrueMedia normalizes to a wider reference (~1.66 ft full width). **This needs empirical validation** — overlay converted TrueMedia locations against your own Trackman locations for the same games (Davidson games should appear in both sources) and check alignment.

**What TrueMedia does NOT provide (confirmed by code inspection):**
- Base occupancy per pitch (no `on_1b/on_2b/on_3b` fields in `GamePitchesTrackman`)
- Running score per pitch (no `home_score/away_score`)
- Player-level pitch data without team context (transfer player gap)
- Pitcher-level pitch-by-pitch usage by count (only aggregate count splits)

### 1.3 Existing Analytics Engine (What's Already Built)

Your codebase is substantial. Here's what's already implemented and relevant to the decision engine:

**Scoring & Ranking:**
- `_pitch_score_composite()` — 22-factor weighted composite (0-100) scoring each pitch type vs a specific hitter. Factors include: Stuff+, whiff%, CSW%, 2-strike whiff rate by pitch class, chase exploitation, tunnel score, EV against, K-prone factor, platoon, wOBA split, contact%, IVB, EffVelo, usage, extension, zone exploitation, barrel% against, horizontal break, InZoneSwing%.
- `_score_pitcher_vs_hitter()` — wraps the composite into a full matchup assessment with per-pitch scores, recommendations, and a usage-weighted overall score.
- `_score_hitter_vs_pitcher()` — the inverse view with pitch-edge ratings.

**Sequencing:**
- `_build_3pitch_sequences()` — hitter-aware 3-pitch sequence builder (setup → bridge → putaway) using tunnel scores, sequence whiff/chase data, effective velo gaps, and putaway vulnerability ranking. Already in both `scouting.py` and `sequencing.py`.
- `_build_4pitch_sequence()` — extension for 4-pitch patterns.
- `_top_tunnel_pairs()` — tunnel-scored pitch pair recommendations.

**At-Bat Scripting:**
- `_generate_at_bat_script()` — generates complete game-plan with tag line, first-pitch plan, ahead plan, 2-strike plan, and pitch edges per situation. This is close to what the in-game engine needs to output.

**Analytics:**
- `_compute_stuff_plus()` — Stuff+ model
- `_compute_command_plus()` — Command+ model
- `_compute_tunnel_score()` — tunnel deception scoring
- `compute_swing_path_metrics()` — attack angle, bat speed proxy, contact depth, per-pitch-type swing path
- `_swing_hole_finder()` — identifies hitter-specific zone vulnerabilities

**Defense:**
- `_recommend_fielder_positions()` — spray-based positioning recommendations
- Player positioning CSV parsing with actual fielder coordinates at release

**Catcher:**
- Framing rates, pop time distribution, per-pitcher receiving performance

---

## Part 2: Data Gaps and How to Close Them

### Gap 1: Base State (Critical for v2, Useful for v1)

**Problem:** Neither your parquet nor the TrueMedia `GamePitchesTrackman` response contains `on_1b/on_2b/on_3b` or `home_score/away_score`. Without this, you cannot compute Run Expectancy (RE) deltas or Win Probability (WP) changes.

**Solution — Reconstruct from play-by-play:**
You have `PlayResult`, `RunsScored`, `OutsOnPlay`, `Inning`, `Outs`, and `Top/Bottom` (in the raw CSV, not the parquet). Base state can be reconstructed:

1. For each game, sort by `PitchNo` ascending
2. Initialize state: `{bases: [0,0,0], outs: 0, score: [0,0]}`
3. On each PA-ending event, apply `PlayResult` rules:
   - Single: runners advance (1 base default, +1 if runner on 2B or 3B scores based on `RunsScored`)
   - Double/Triple/HR: standard advancement
   - Out: `OutsOnPlay` outs added, runner advancement based on out type
   - SB/CS: specific runner movement

This reconstruction will be ~90% accurate. The 10% error comes from runner-advancement ambiguity (e.g., single with runner on first — did they go to second or third?). For a decision engine, this is good enough.

**Deliverable:** A `reconstruct_base_state()` function that takes a game's pitch DataFrame (sorted by PitchNo) and returns base/score state for each pitch. Store as new columns: `on_1b`, `on_2b`, `on_3b`, `home_score`, `away_score`.

**Effort:** ~200 lines of Python. Medium complexity, high value.

### Gap 2: Parquet Column Gaps (Easy Fix, High ROI)

**Problem:** Your parquet is missing `EffectiveVelo`, `AutoPitchType`, `Top/Bottom`, `PitchUID`, and other fields that are present in the raw CSVs.

**Solution:** Update your parquet build pipeline to include these columns. This is a one-time configuration change wherever you run the CSV → parquet conversion. Add to the SELECT list:
- `EffectiveVelo`, `AutoPitchType`, `Top/Bottom`, `Catcher`, `CatcherTeam`
- `PitchUID`, `PlayID` (for joining positioning data)
- `PopTime`, `ExchangeTime`, `HangTime`, `ZoneSpeed`
- `PitchReleaseConfidence`, `PitchLocationConfidence` (for quality filtering)
- `ContactPositionX/Y/Z`, `PositionAt110X/Y/Z` (for advanced spray and contact depth)

**Effort:** Trivial (column list change in ETL). Re-process existing CSVs.

### Gap 3: Pitch Type Reconciliation (Medium Priority)

**Problem:** Your parquet has 7,198 `Undefined` pitch types (5.1%) and 138 `Other`. The `AutoPitchType` (Trackman's ML classifier) is not available in the parquet. TrueMedia and Trackman may classify the same pitch differently.

**Solution (3 parts):**
1. Add `AutoPitchType` to parquet (Gap 2 fix)
2. For Davidson pitchers: compare `TaggedPitchType` vs `AutoPitchType` agreement rate. Flag pitchers where a pitch type has <80% agreement (likely misclassification). Build a reconciliation map per pitcher.
3. For TrueMedia opposition data: when merging TM pitch-level data with any Trackman expectations, use TrueMedia's classification as primary (since that's what the hitter saw) but flag discrepancies.

**Effort:** ~100 lines of Python for the comparison. Ongoing vigilance.

### Gap 4: Transfer Player History (Acknowledged Limitation)

**Problem:** The TrueMedia API requires `teamId` for `PlayerTotals` and `TeamGames`. A player who transferred to Bryant for 2026 has their 2024 data under a different team ID. `GamePitchesTrackman` requires game IDs, not player IDs.

**Solution for v1:** Use aggregate `PlayerTotals` from the player's current team for the current season. Accept that pitch-level data is unavailable for prior-team seasons unless you manually identify and fetch those team's game IDs.

**Solution for v2:** Build a player ID crosswalk. TrueMedia returns `playerId` in `PlayerTotals`. If you can query `PlayerTotals` with a player filter (test: `filters=(event.playerId={id})` without teamId), you get cross-team aggregates. For pitch-level, you'd need to enumerate every team the player played for and fetch those games.

**Effort:** v1 is zero (accept limitation). v2 is moderate (player-centric data pull pipeline).

### Gap 5: NCAA-Specific Run Values (Must Define)

**Problem:** The original plan says "NCAA weights" for linear run values but doesn't specify them. MLB weights are wrong for college baseball (BBCOR bats produce different BABIP, HR rates, and run environment).

**Solution:** Derive NCAA D1 linear weights from your own data:
1. From the reconstructed base states (Gap 1), compute RE288 table (8 base states × 3 out states × 12 situations = 288 cells, or the standard 24-cell RE matrix)
2. For each event type (1B, 2B, 3B, HR, BB, HBP, K, Out), compute the average RE change
3. These become your linear weights for v1

**Approximate NCAA D1 weights (from public research, verify against your data):**

| Event | MLB Weight | Estimated NCAA D1 Weight |
|---|---|---|
| BB/HBP | 0.69 | ~0.65 |
| 1B | 0.88 | ~0.82 |
| 2B | 1.24 | ~1.18 |
| 3B | 1.56 | ~1.48 |
| HR | 2.01 | ~1.85 |
| Out | -0.25 | ~-0.23 |
| K | -0.27 | ~-0.25 |

**Effort:** ~150 lines of Python once base state reconstruction exists. High importance — every scoring decision flows through these weights.

---

## Part 3: Architecture

### 3.1 System Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    IN-GAME UI (Streamlit)                      │
│  Coach inputs: Pitcher, Batter, Count, Outs, Inning,          │
│                Base State, Score (manual or auto)              │
├──────────────────────────────────────────────────────────────┤
│                    DECISION ENGINE CORE                        │
│                                                                │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │ Pitch Call   │  │ Location     │  │ Baserunning/       │   │
│  │ Recommender  │  │ Recommender  │  │ Defense Recommender │   │
│  └──────┬──────┘  └──────┬───────┘  └────────┬───────────┘   │
│         │                │                    │                │
│  ┌──────▼────────────────▼────────────────────▼──────────┐    │
│  │              MATCHUP SCORER (exists: scouting.py)      │    │
│  │  _pitch_score_composite() + count-state adjustments    │    │
│  └───────────────────────┬───────────────────────────────┘    │
│                          │                                     │
├──────────────────────────▼─────────────────────────────────┤
│                   PRE-COMPUTED DATA LAYER                      │
│                                                                │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐    │
│  │ Davidson       │  │ Opponent Pack  │  │ D1 Priors    │    │
│  │ Pitcher        │  │ (Bryant 2024-  │  │ (league-wide │    │
│  │ Arsenals       │  │  2026)         │  │  baselines)  │    │
│  └────────────────┘  └────────────────┘  └──────────────┘    │
│                                                                │
│  Sources: Trackman parquet + DuckDB  |  TrueMedia API         │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Module Structure

```
decision_engine/
├── __init__.py
├── core/
│   ├── matchup.py          # Wraps _pitch_score_composite with count/base awareness
│   ├── count_value.py      # Count transition values (RE cost of ball/strike)
│   ├── linear_weights.py   # NCAA D1 run values (computed or hardcoded)
│   ├── shrinkage.py        # Bayesian shrinkage to D1 priors
│   └── state.py            # Game state management (count, base, score, outs, inning)
├── recommenders/
│   ├── pitch_call.py       # "What pitch to throw" — top recommendation + 2 alternates
│   ├── pitch_location.py   # "Where to locate" — zone target + miss band
│   ├── baserunning.py      # Steal/hold/pick risk (uses catcher pop time + pitcher delivery)
│   └── defense.py          # Positioning overlay (wraps existing defense.py)
├── data/
│   ├── opponent_pack.py    # Pre-fetch and cache Bryant (or any team) data
│   ├── base_state.py       # Reconstruct base occupancy from play-by-play
│   ├── priors.py           # D1 population baselines (from your DuckDB population queries)
│   └── pitch_reconcile.py  # AutoPitchType vs TaggedPitchType reconciliation
├── simulation/
│   ├── count_sim.py        # v1: simulate next 1-3 pitches via count transitions
│   └── re_sim.py           # v2: full RE/WP tree (requires base state)
└── ui/
    ├── ingame_panel.py     # Streamlit page for in-game use
    └── pregame_panel.py    # Streamlit page for pre-game plan review
```

### 3.3 Key Design Decisions

**Decision 1: Extend `_pitch_score_composite`, don't replace it.**
Your existing 22-factor scorer is well-tuned. The decision engine adds count awareness and base-state adjustments on top, not a competing model. Specifically:

```python
def recommend_pitch(state, arsenal, hitter_profile, d1_priors):
    """
    state: {count: (balls, strikes), outs, inning, bases: [0,1,0], score: (3,2)}
    arsenal: output of _get_our_pitcher_arsenal()
    hitter_profile: output of _get_opp_hitter_profile()
    d1_priors: population baselines for shrinkage
    """
    # Step 1: Base composite score (existing)
    matchup = _score_pitcher_vs_hitter(arsenal, hitter_profile)
    
    # Step 2: Count adjustment
    #   - In 0-2 or 1-2: boost putaway pitches, penalize hittable fastballs
    #   - In 2-0 or 3-1: boost command pitches, penalize wild secondaries
    #   - Weight by count_transition_value (cost of next ball vs benefit of next strike)
    count_adj = compute_count_adjustments(state.count, matchup.pitch_scores)
    
    # Step 3: Base-state adjustment (v1: coarse rules)
    #   - Runner on 3B, <2 outs → boost GB-inducing pitches
    #   - Bases empty → maximize K probability
    #   - Runner on 1B, <2 outs → consider double-play pitch (sinker down)
    base_adj = compute_base_adjustments(state.bases, state.outs, matchup.pitch_scores)
    
    # Step 4: Shrinkage (critical for small samples)
    #   - If pitcher has <50 pitches of a type, blend toward D1 average
    #   - If hitter has <30 PA, blend toward D1 average for their profile
    shrunk_scores = apply_shrinkage(matchup.pitch_scores, d1_priors, arsenal, hitter_profile)
    
    # Step 5: Combine and rank
    final_scores = combine_adjustments(shrunk_scores, count_adj, base_adj)
    return rank_recommendations(final_scores, top_n=3)
```

**Decision 2: Count-transition values as first-class input.**
For each count state, pre-compute:
- `value_of_strike(count)` = RE(next count after strike) - RE(current count)
- `value_of_ball(count)` = RE(next count after ball) - RE(current count)
- `leverage = value_of_strike - value_of_ball` (how much the count matters right now)

These values determine how aggressive vs conservative the recommendation should be. In 3-0, the cost of a ball (walk) is huge, so recommend the highest-command pitch even if it has lower stuff. In 0-2, the cost of a ball is low, so recommend the best putaway even if it's wild.

**Decision 3: Shrinkage is the most important modeling component.**
At the college level, a pitcher might have 47 sliders all season. A hitter might have 12 PA against LHP. The engine must blend toward population priors aggressively:

```python
def shrunk_estimate(observed, n_obs, prior, n_prior_equiv=50):
    """
    observed: sample stat (e.g., whiff rate 40%)
    n_obs: sample size (e.g., 47 pitches)
    prior: D1 population average (e.g., 28% whiff on sliders)
    n_prior_equiv: how many observations the prior is "worth"
    """
    weight = n_obs / (n_obs + n_prior_equiv)
    return weight * observed + (1 - weight) * prior
```

With `n_prior_equiv=50`, a pitcher with 47 sliders gets roughly 50/50 blend between their data and the D1 average. With 200 sliders, they're 80% their own data. This is critical — without it, the engine will make wild recommendations based on tiny samples.

**Decision 4: Location recommendations use Command+ as a Gaussian blur.**
Your existing `_compute_command_plus()` gives a per-pitch-type command rating. Convert this into a concrete location recommendation:

1. Identify the "ideal" target zone from the matchup analysis (e.g., "low-away slider")
2. Model the pitcher's actual location distribution for that pitch type as a 2D Gaussian (mean = their typical location, variance = their historical spread for that pitch type)
3. The recommendation becomes: "Target zone X, expected to land in zone Y with Z% probability"
4. If the pitcher's spread is too wide to reliably hit the target, recommend a zone they *can* hit that's still effective

This prevents recommending "backdoor slider on the corner" to a pitcher who misses by 8 inches on average.

---

## Part 4: Implementation Phases

### Phase 0: Data Foundation (Week 1-2)

**0a. Expand parquet pipeline**
- Add 15+ columns from raw CSV to parquet (list in Gap 2)
- Re-process all existing CSVs
- Update `loader.py` view to include new columns

**0b. Build base-state reconstruction**
- New module: `decision_engine/data/base_state.py`
- Input: game DataFrame sorted by PitchNo
- Output: per-pitch `on_1b`, `on_2b`, `on_3b`, `home_score`, `away_score`
- Validation: spot-check 20 games manually

**0c. Compute NCAA D1 run environment**
- RE24 matrix from reconstructed base states
- Linear weights per event type
- Count-transition values for all 12 count states

**0d. Pitch type reconciliation**
- Compare `AutoPitchType` vs `TaggedPitchType` for all Davidson pitchers
- Build per-pitcher reconciliation map
- Flag problematic classifications (>20% disagreement)

**Deliverable:** Updated parquet, base-state table, RE24 matrix, count-transition values, pitch-type reconciliation report.

### Phase 1: Core Decision Engine — v1 (Week 3-5)

**1a. Opponent Pack Builder**
- New module: `decision_engine/data/opponent_pack.py`
- For a given team + seasons, fetch and cache:
  - `PlayerTotals` (hitting + pitching, raw and vs-hand splits)
  - `GamePitchesTrackman` for all games
  - Count splits via `fetch_hitter_count_splits`
- Normalize all data using existing `_normalize_tm_df` / `_TM_TO_TRACKMAN_COLS` pipeline
- Store as DuckDB tables (reuse your existing `davidson.duckdb` pattern)
- Include staleness tracking (when was data last refreshed)

**1b. D1 Prior Baselines**
- Compute population baselines from your full D1 parquet:
  - Per-pitch-type whiff%, chase%, CSW%, EV against, barrel% (by pitcher hand × batter hand)
  - Per-count zone swing%, contact%, damage rates
  - Per-pitch-type stuff+ and command+ distributions
- Store as a small lookup table (~200 rows)
- These become the shrinkage targets

**1c. Shrinkage Module**
- Implement Bayesian shrinkage for all key metrics
- Auto-detect sample size from pitch count or PA count
- Expose confidence level to UI ("high confidence" = >100 observations, "low confidence, blended with D1 avg" = <30)

**1d. Count-Aware Pitch Recommender**
- Wrap `_pitch_score_composite` with:
  - Count-transition value adjustments
  - Count-specific hitter tendencies (from count splits)
  - "Waste pitch" logic in 0-2 counts (tunnel setup for putaway)
  - "Must-strike" logic in 3-x counts (prioritize command over stuff)
- Output: top 3 pitch recommendations with scores, reasons, and confidence levels

**1e. Location Recommender**
- For each recommended pitch type:
  - Compute "ideal zone" from hitter vulnerability × pitch design physics (PZM from your `_ze_pzm` logic)
  - Apply command blur (pitcher's historical location variance for that pitch type)
  - Output: primary target zone, acceptable miss zone, danger zone
- Visual: 2D heatmap overlay on strike zone

**1f. Base-State Coarse Adjustments (v1)**
- Manual input of bases/outs (dropdown in UI)
- Rule-based adjustments:

| Situation | Adjustment |
|---|---|
| Runner 3B, <2 outs | +15 to GB-inducing pitches, -10 to FB pitches |
| Bases empty, 2 outs | +10 to K pitches, neutral otherwise |
| Runner 1B, <2 outs | +10 to sinkers/GB, +5 to curveballs (DP ball) |
| Bases loaded | +20 to highest-K pitch, -15 to walk-risk pitches |
| Runners in scoring, 2 outs | Maximize K probability regardless |

**Deliverable:** Working pitch call + location recommender. Runnable in Streamlit. Handles any Davidson pitcher vs any opponent hitter.

### Phase 2: Baserunning & Defense Integration (Week 5-6)

**2a. Baserunning Recommender**
- Input: runner speed (from TrueMedia speed score or manual), pitcher delivery time (from Trackman `ZoneTime` or estimated from `Extension` + `RelSpeed`), catcher pop time (from `catcher.py` data)
- Output: steal green/yellow/red light with probability of success
- Include pitch-type dependency (breaking ball = longer delivery = better steal opportunity)

**2b. Defense Positioning**
- Extend existing `_recommend_fielder_positions()` to be matchup-specific:
  - Input: hitter spray chart + pull/oppo/center tendencies + pitch call recommendation
  - Output: fielder positioning shifts per pitch type being thrown
- Integrate player positioning CSV data (actual vs recommended comparison)

**Deliverable:** Baserunning traffic light + defensive positioning recommendations layered onto pitch call.

### Phase 3: Simulation & RE/WP (Week 7-9) — v2

**3a. Count Simulation (v1 extension)**
- Given current count + recommended pitch, simulate:
  - P(strike | pitch_type, zone, batter_hand) from your data
  - P(ball | same)
  - P(swing | same) × P(whiff | swing) × P(foul | contact) × P(in-play | contact)
  - Recurse for 1-3 pitches forward
- Output: expected PA outcome distribution (K%, BB%, InPlay%, with damage estimate)
- Use for "what if I throw the #2 recommendation instead?" comparison

**3b. Full RE/WP Tree (requires base state)**
- Only possible after Phase 0b is complete and validated
- For each pitch decision, compute:
  - `Value(pitch|state) = Σ P(outcome|pitch,state) × RE(next_state)`
  - Where outcomes include: ball, called strike, swinging strike, foul, in-play (by type)
  - And state transitions update count, base occupancy, outs, score
- WP mode when score/inning are available:
  - `WP_delta(pitch|state) = Σ P(outcome|pitch,state) × WP(next_state) - WP(current_state)`
- This replaces the linear-weights proxy with true optimality

**Deliverable:** Simulate PA button, RE delta display, WP overlay when score is known.

### Phase 4: Validation & Calibration (Ongoing, starting Week 4)

**4a. Holdout Backtesting**
- Split 2025 Davidson data temporally: first 75% of games for training, last 25% for testing
- For each pitch in the test set, compute what the engine would have recommended
- Grade: did the recommended pitch type produce better outcomes than the actual pitch thrown?
- Metrics: recommended-pitch whiff rate vs actual, recommended-pitch EV against vs actual

**4b. Calibration Plots**
- For each pitch type × zone × count:
  - Predicted whiff% vs observed whiff% (should be a 45-degree line)
  - Predicted damage (EV/barrel) vs observed
- Identify miscalibrated regions and adjust shrinkage weights

**4c. Shadow Mode**
- Run engine during live games without showing coaches
- After each game: "what would the engine have recommended vs what actually happened"
- Grade retroactively: did the engine's recommendations correlate with better outcomes?

**4d. Confidence Gating**
- Display confidence tier with every recommendation:
  - **High** (>100 pitcher observations for that pitch type, >50 hitter PA): "Engine recommendation"
  - **Medium** (50-100 / 20-50): "Blended with D1 averages"
  - **Low** (<50 / <20): "Mostly D1 priors — use coaching judgment"
- Never display a recommendation without a confidence indicator

---

## Part 5: In-Game UI Specification

### 5.1 Input Panel (Left Side)

```
┌─────────────────────────────────┐
│ OUR PITCHER: [dropdown]         │
│ THEIR HITTER: [dropdown]        │
│                                 │
│ COUNT:  ○○○ Balls  ○○ Strikes   │
│ OUTS:   ○○○                     │
│ INNING: [1-9+] [Top/Bot]       │
│                                 │
│ BASES:  ◇ (clickable diamond)   │
│ SCORE:  DAV [__] OPP [__]      │
│                                 │
│ [Auto-advance count ☑]          │
│ [RESET PA] [NEW BATTER]        │
└─────────────────────────────────┘
```

The hitter dropdown auto-populates from the pre-fetched opponent pack. Count auto-advances if the checkbox is on (after each pitch, coach taps the outcome: ball/strike/foul/in-play).

### 5.2 Recommendation Panel (Center)

```
┌────────────────────────────────────────────┐
│ PITCH CALL                    Confidence: ██████░░ High │
│                                                         │
│  #1  SLIDER  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░  78                  │
│       Low-away, expand off plate                        │
│       Hitter chases 34% + our 42% whiff                │
│                                                         │
│  #2  FASTBALL ▓▓▓▓▓▓▓▓▓▓▓▓░░░░░  64                  │
│       Up-and-in, use IVB advantage (17")                │
│                                                         │
│  #3  CHANGEUP ▓▓▓▓▓▓▓▓▓▓░░░░░░░  58                  │
│       Arm-side down, tunnel off FB                      │
│                                                         │
├────────────────────────────────────────────┤
│ LOCATION TARGET        PITCH OUTCOME DIST               │
│                                                         │
│  [strike zone heatmap]  K:  28%                         │
│  [with target dot and   BB:  6%                         │
│   miss probability      IP: 42% (EV ~84 mph)           │
│   contour]              Foul: 24%                       │
│                                                         │
├────────────────────────────────────────────┤
│ SEQUENCE PLAN (if early in PA)                          │
│  FB up → SL down-away → SL chase (tunnel: A)           │
│                                                         │
│ BASERUNNING: Hold (pop time 1.95s, steal prob 31%)     │
│ DEFENSE: Shade pull-side, IF normal depth               │
└────────────────────────────────────────────┘
```

### 5.3 Latency Requirements

| Operation | Target | How |
|---|---|---|
| Pitcher/batter selection → first recommendation | <1 second | Pre-computed arsenals + cached opponent packs |
| Count change → updated recommendation | <200ms | Count adjustments are arithmetic on cached scores |
| Base state change → updated recommendation | <200ms | Rule-based adjustments, no re-computation |
| "Simulate PA" button → result | <2 seconds | Monte Carlo with 1000 iterations on cached probabilities |

All heavy computation (Stuff+, Command+, tunnel scores, composite scoring) happens at opponent-pack build time or pitcher-arsenal build time, not at query time.

---

## Part 6: Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Small sample sizes dominate most recommendations | **Critical** | Aggressive shrinkage to D1 priors; always show confidence tier; never recommend based on <10 observations without explicit "low confidence" flag |
| Coordinate system mismatch between TrueMedia and Trackman | **High** | Validate with overlapping Davidson games; scatter plot comparison before any location recommendations go live |
| Pitch type misclassification | **Medium** | Add AutoPitchType to parquet; build reconciliation layer; flag >20% disagreement pitchers |
| Latency too high for in-game use | **Medium** | Pre-compute everything possible; cache at opponent-pack build time; UI shows stale but fast data, background refreshes |
| Coach trust collapse after single bad recommendation | **High** | Always show outcome distribution, not just top pick; frame as "decision support" not "decision maker"; shadow-test before deployment |
| Transfer players with no pitch-level history | **Medium** | Fall back to aggregate totals + D1 priors; flag "limited data" explicitly |
| TrueMedia API downtime during game | **Medium** | All opponent data pre-fetched and cached locally; engine works fully offline from cache |
| Base state reconstruction errors compound | **Low** | Validate against 20+ known games; accept ~90% accuracy for v2 RE calculations; v1 uses manual input |

---

## Part 7: Open Questions for You

1. **Seasons for Bryant:** Which seasons should the opponent pack cover? `2024+2025`, `2025+2026 YTD`, or `2024+2025+2026`?

2. **Transfer players:** For v1, is aggregate-only acceptable for players who transferred to Bryant, or do you need pitch-level history from their prior school?

3. **RE vs WP objective:** Do you want:
   - RE-based recommendations always (optimize run scoring/prevention)
   - WP-based when score/inning are available, RE otherwise (optimize win probability, which changes late-game strategy)
   - Coach-selectable toggle

4. **Base state data source:** Can you confirm whether TrueMedia's `GamePitchesTrackman` response includes any base-state or score fields you haven't mapped yet? (Check raw JSON response for fields like `runnersOn`, `homeScore`, `awayScore`, `baseState`)

5. **Deployment environment:** Is this running on a laptop in the dugout, a tablet, or a phone? This affects UI layout and latency constraints.

6. **Who operates it during games?** A student manager? An analytics staff member? The pitching coach? This affects how much context the UI needs to provide vs assume.

---

## Part 8: Recommended Build Order

**Start here (highest ROI, lowest risk):**
1. Expand parquet columns (Phase 0a) — trivial, unlocks everything else
2. Build opponent pack for Bryant (Phase 1a) — uses existing API code, validates data pipeline
3. Implement shrinkage module (Phase 1c) — the single most impactful modeling decision
4. Build count-aware pitch recommender (Phase 1d) — extends your existing scorer with count intelligence
5. Build in-game UI shell (Phase 5) — even with v0.5 quality recommendations, getting it in coaches' hands for feedback is essential

**Then iterate:**
6. Location recommender (Phase 1e)
7. Base state reconstruction (Phase 0b)
8. Validation/calibration (Phase 4)
9. Baserunning/defense integration (Phase 2)
10. Full RE/WP simulation (Phase 3)

Each phase is independently valuable. You can ship after step 5 and improve incrementally.
