# Davidson Baseball Audit Report

Generated: 2026-02-05T01:43:49+00:00
Parquet: `/Users/ahanjain/davidson_baseball/all_trackman.parquet`
DuckDB: `/Users/ahanjain/davidson_baseball/davidson.duckdb`

## Meta Fingerprint
| Field | Parquet | DB Meta | Match |
|---|---|---|---|
| path | /Users/ahanjain/davidson_baseball/all_trackman.parquet | /Users/ahanjain/davidson_baseball/all_trackman.parquet | YES |
| mtime | 1770062504.8447099 | 1770062504.8447099 | YES |
| size | 732144069 | 732144069 | YES |

## Required Tables
| Table | Present |
|---|---|
| trackman_pop | YES |
| batter_stats_pop | YES |
| pitcher_stats_pop | YES |
| stuff_baselines | YES |
| tunnel_population | YES |
| meta | YES |

## Row Counts
- trackman_pop rows: 3,708,551
- trackman_parquet rows: 3,708,551

## Outlier Checks (trackman_pop)
| Check | Count |
|---|---|
| PlateLocSide out of range | 52,757 |
| PlateLocHeight out of range | 59,939 |
| Direction out of range (InPlay) | 5,750 |
| Distance out of range (InPlay) | 10 |

## Tunnel Kinematics Sanity (9-param)
- Sample rows: 2,000
- Valid 9-param solutions: 2,000 (100.0%)
- PlateLocSide MAE (current sign): 0.0455 ft
- PlateLocSide MAE (flipped sign): 1.6171 ft
- PlateLocHeight MAE: 0.1713 ft
- NOTE: Current sign flip aligns better with PlateLocSide.
- Flight time t_total (s): min 0.364, p5 0.389, p50 0.423, p95 0.484, max 0.563

## Population Parity (DB vs Parquet)
- Batters: ok (mismatch rows: 0)
- Pitchers: ok (mismatch rows: 0)

## Verdict
PASSED
