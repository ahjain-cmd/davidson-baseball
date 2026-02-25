# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Davidson College baseball analytics platform. Streamlit web app backed by DuckDB (precomputed from Trackman pitch-level parquet data) with TrueMedia API integration for opponent scouting. Deployed via Docker on a DigitalOcean droplet at wildcatsdb.com.

**Team ID:** `DAV_WIL` (Davidson's Trackman identifier — used everywhere as `DAVIDSON_TEAM_ID`)

## Key Commands

```bash
# Run locally
streamlit run app.py

# Rebuild DuckDB from parquet (required after adding new game data)
python3 precompute.py --parquet all_trackman.parquet --out davidson.duckdb --overwrite

# Deploy to server (wildcatsdb.com)
git push origin main
ssh -i ~/ahanjainndavidsonbaseball root@165.227.182.92
cd ~/davidson-baseball && git pull && docker compose down && docker compose up -d --build
# SSH key passphrase: davbas

# If parquet data changed, also upload and rebuild on server:
scp -i ~/ahanjainndavidsonbaseball all_trackman.parquet root@165.227.182.92:~/davidson-baseball/
# Then SSH in, run precompute inside Docker (entrypoint override):
docker run --rm --entrypoint python3 \
  -v $(pwd)/all_trackman.parquet:/app/all_trackman.parquet:ro \
  -v $(pwd):/output \
  davidson-baseball-app:latest \
  precompute.py --parquet /app/all_trackman.parquet --out /output/davidson.duckdb --overwrite
# Then restart: docker compose down && docker compose up -d --build
```

## Data Pipeline

```
Raw Trackman CSVs (v3/{year}/{month}/{day}/CSV/)
  → merged into all_trackman.parquet (~700MB, string Date column "YYYY-MM-DD")
    → precompute.py builds davidson.duckdb with:
        trackman (VIEW) - normalized names, pitch types, season calc, deduped
        davidson_data (TABLE) - Davidson-involved games only (~230K rows)
        batter_stats_pop / pitcher_stats_pop - pre-aggregated season stats
        stuff_baselines - population means/stds for Stuff+ z-scores
        stuff_plus - per-pitch Stuff+ values
        tunnel_population / tunnel_pair_outcomes - tunnel scoring data
        sidebar_stats / seasons / meta - app metadata
```

**Critical:** The `Date` column in parquet is VARCHAR not DATE. Any DuckDB SQL using `EXTRACT(MONTH FROM "Date")` must cast first: `EXTRACT(MONTH FROM CAST("Date" AS DATE))`.

**Docker mounts:** `davidson.duckdb` (read-only), `.cache/` directory, `export.json` (optional). The parquet is NOT mounted — all data comes through DuckDB.

## Adding New Games from Trackman CSVs

Each game comes as one or two files from Trackman:
- `{date}-{venue}-{game#}_unverified.csv` — pitch-level data (167 cols) **(required)**
- `{date}-{venue}-{game#}_unverified_playerpositioning_FHC.csv` — fielder positioning (30 extra cols) **(optional)**

**Critical:** `config.py` resolves `PARQUET_PATH` to `all_trackman_fixed.parquet` if it exists, otherwise `all_trackman.parquet`. Always merge into `all_trackman_fixed.parquet` — that's what precompute reads.

### Steps

#### 1. Merge CSV into the parquet using DuckDB

The full script below handles both cases (with or without FHC). It auto-detects type mismatches between CSV and parquet and casts accordingly. Run from `~/davidson_baseball/`.

```python
import duckdb
con = duckdb.connect()

csv_path = "<trackman_csv>"           # e.g. "/Users/ahanjain/20260224-WoffordCollege-1_unverified.csv"
fhc_path = "<fhc_csv_or_None>"       # e.g. None if no FHC file
parquet_path = "all_trackman_fixed.parquet"
out_path = "all_trackman_fixed_new.parquet"

# If FHC file exists, join it with the trackman CSV first
if fhc_path:
    positioning_cols = [
        'DetectedShift',
        '1B_PositionAtReleaseX', '1B_PositionAtReleaseZ',
        '2B_PositionAtReleaseX', '2B_PositionAtReleaseZ',
        '3B_PositionAtReleaseX', '3B_PositionAtReleaseZ',
        'SS_PositionAtReleaseX', 'SS_PositionAtReleaseZ',
        'LF_PositionAtReleaseX', 'LF_PositionAtReleaseZ',
        'CF_PositionAtReleaseX', 'CF_PositionAtReleaseZ',
        'RF_PositionAtReleaseX', 'RF_PositionAtReleaseZ',
        '1B_Name', '1B_Id', '2B_Name', '2B_Id',
        '3B_Name', '3B_Id', 'SS_Name', 'SS_Id',
        'LF_Name', 'LF_Id', 'CF_Name', 'CF_Id',
        'RF_Name', 'RF_Id', 'FHC'
    ]
    fhc_select = ", ".join(f'fhc."{c}"' for c in positioning_cols)
    csv_source = f"""(
        SELECT tm.*, {fhc_select}
        FROM read_csv('{csv_path}', auto_detect=true) tm
        LEFT JOIN read_csv('{fhc_path}', auto_detect=true) fhc
        ON tm."PitchNo" = fhc."PitchNo" AND tm."Date" = fhc."Date"
    )"""
else:
    csv_source = f"read_csv('{csv_path}', auto_detect=true)"

# Get schemas
parquet_cols = con.execute(f"SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM read_parquet('{parquet_path}'))").fetchall()
csv_cols = con.execute(f"SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM {csv_source})").fetchall()
csv_names = {c for c, _ in csv_cols}
parquet_types = {c: t for c, t in parquet_cols}

# Build casted SELECT: cast mismatched types, add NULLs for missing cols
csv_select_parts = []
for c, t in csv_cols:
    if c in parquet_types and parquet_types[c] != t:
        csv_select_parts.append(f'CAST("{c}" AS {parquet_types[c]}) AS "{c}"')
    else:
        csv_select_parts.append(f'"{c}"')
for c, t in parquet_cols:
    if c not in csv_names:
        csv_select_parts.append(f'NULL::{t} AS "{c}"')

csv_select = ", ".join(csv_select_parts)
parquet_col_order = ", ".join(f'"{c}"' for c, _ in parquet_cols)

# Merge: UNION ALL existing + new → write new parquet
con.execute(f"""
    COPY (
        SELECT {parquet_col_order} FROM read_parquet('{parquet_path}')
        UNION ALL
        SELECT {parquet_col_order} FROM (SELECT {csv_select} FROM {csv_source})
    ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
""")

# Verify row counts
old = con.execute(f"SELECT COUNT(*) FROM read_parquet('{parquet_path}')").fetchone()[0]
new = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]
csv_n = con.execute(f"SELECT COUNT(*) FROM {csv_source}").fetchone()[0]
print(f"Old: {old:,} + CSV: {csv_n:,} = New: {new:,}")
assert new == old + csv_n, "Row count mismatch!"
```

**Common type mismatches:** `PitcherId`, `BatterId`, `HomeTeamForeignID`, `AwayTeamForeignID` (BIGINT→VARCHAR), `y0` (BIGINT→DOUBLE). The script handles these automatically.

#### 2. Swap parquet files
```bash
mv all_trackman_fixed.parquet all_trackman_fixed.parquet.bak
mv all_trackman_fixed_new.parquet all_trackman_fixed.parquet
```

#### 3. Rebuild DuckDB (must use --overwrite)
```bash
python3 precompute.py --overwrite
```
This rebuilds `davidson.duckdb` and exports `.cache/davidson_data.feather`.

#### 4. Upload to server and deploy
```bash
# Upload all three files (parquet, duckdb, feather)
scp -i ~/ahanjainndavidsonbaseball all_trackman_fixed.parquet root@165.227.182.92:~/davidson-baseball/
scp -i ~/ahanjainndavidsonbaseball davidson.duckdb root@165.227.182.92:~/davidson-baseball/
scp -i ~/ahanjainndavidsonbaseball .cache/davidson_data.feather root@165.227.182.92:~/davidson-baseball/.cache/
# SSH key passphrase: davbas

# Restart the container
ssh -i ~/ahanjainndavidsonbaseball root@165.227.182.92 \
  "cd ~/davidson-baseball && docker compose down && docker compose up -d --build"
```

**Troubleshooting:** If Docker build fails with snapshot errors, SSH in and run `docker system prune -f` before rebuilding.

## Architecture

### App Routing (app.py)
Sidebar radio selector routes to page functions. Each page receives `data` (full Davidson DataFrame from `load_davidson_data()`).

### Pages (_pages/)
- **postgame.py** (~2800 lines): Largest file. Pitcher grades (Stuff+/Command+), hitter grades, call grade system, best hitting zones, at-bat review. Most new feature work happens here.
- **scouting.py** (~12K lines): Opponent scouting with TrueMedia API. `_pitch_score_composite()` is the 22-factor scoring engine.
- **pitching.py**: Pitcher cards, Stuff+/Command+ bars, tunnel pairs, `_rank_pairs()` and `_rank_sequences_from_pdf()` are reused by postgame.
- **hitting.py**: Hitter cards, zone profiles, swing decision lab.

### Analytics (analytics/)
- **stuff_plus.py**: Z-score composite across 6 metrics per pitch type (velo, break, extension, spin, VAA). Scale: 0-200, 100=average.
- **tunnel.py**: Physics-based pitch deception — Euclidean distance in multi-dimensional space, percentile-normalized.
- **command_plus.py**: Location quality metric + pitch pair result tables.
- **zone_vulnerability.py**: 3×3 zone grids — `compute_zone_swing_metrics()`, `compute_hole_scores_3x3()`, `analyze_zone_patterns()`. Used by scouting, postgame, decision engine.
- **run_expectancy.py** / **win_probability.py**: RE24 and WP models from Trackman data. Cache with parquet fingerprint invalidation.

### Decision Engine (decision_engine/)
In-game pitch call recommender. Bayesian shrinkage for small samples, count-group multipliers, zone vulnerability overlays. Entry point: `ui/ingame_panel.py` → `render_ingame_panel()`.

### PDF Generators (generate_*.py)
All matplotlib-based, landscape 11×8.5. Import shared helpers from `generate_postgame_report_pdf.py` (`_header_bar`, `_styled_table`, `_mpl_call_grade_box`, etc.) and reuse `_compute_call_grade` from `_pages/postgame.py`.

### Data Loading (data/)
- **loader.py**: DuckDB connection (`get_duckdb_con()`), `load_davidson_data()`, `query_population()`. TrueMedia helpers.
- **truemedia_api.py**: TrueMedia REST API client. Converts TM coordinates to Trackman format. 24-hour TTL cache.
- **population.py**: Wraps DuckDB queries for population stats with `@st.cache_data`.
- **bryant_combined.py**: Special multi-year opponent pack builder for Bryant University.

## Conventions

### Name Formats
- Trackman: `"Last, First"` — used in all DataFrames
- TrueMedia: `"First Last"` — API returns this
- Display: `display_name()` converts to `"First Last"` with optional HTML escaping
- Conversion: `tm_name_to_trackman()` and SQL `_name_case_sql()` for normalization

### Pitch Type Normalization
Raw Trackman types are mapped in config.py: `FourSeamFastBall` → `Fastball`, `TwoSeamFastBall/OneSeamFastBall` → `Sinker`, `ChangeUp` → `Changeup`. Types with <5% usage are filtered by `filter_minor_pitches()`.

### Zone Constants
`ZONE_SIDE=0.83`, `ZONE_HEIGHT_BOT=1.5`, `ZONE_HEIGHT_TOP=3.5` (feet). Batter-specific adaptive zones use 5th/95th percentile of called strikes when ≥20 samples available.

### Swing/Contact Classification
`SWING_CALLS`: StrikeSwinging, FoulBall, FoulBallNotFieldable, FoulBallFieldable, InPlay. `CONTACT_CALLS`: excludes StrikeSwinging. Both have SQL tuple versions (`_SWING_CALLS_SQL`).

### Barrel Detection
Statcast formula: EV ≥ 98 mph, launch angle between dynamic bounds (26° at 98 mph, widening to 8°-50° at 116+ mph). Implemented in `is_barrel_mask()`.

### Caching Pattern
Expensive calibrations (RE, WP, pitch priors) use parquet mtime fingerprint — regenerate if data changes. DuckDB tables are rebuilt via `precompute.py --overwrite`.

## Stashed Work

`git stash list` — check for shelved feature branches. As of Feb 2026, `postgame-phase3b-enhancements` contains: enhanced call grade rendering (count strategy, pitch strength, expanded recommendations), best hitter zone improvements (scatter legend, zone patterns, game alignment), and AB review PDF enhancements (scouting fallbacks, inning trend observations).

## Git Remotes

- `origin`: https://github.com/ahjain-cmd/davidson-baseball.git (primary)
- `hf`: HuggingFace Spaces (legacy, may be stale)


For any PDF export functions for scouts or postgame reports, ensure no text, boxes, etc overlap and everything is visually clean. 
