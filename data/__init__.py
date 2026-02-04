"""Data loading, stats computation, and population queries."""
from data.loader import (
    get_duckdb_con, query_population, load_davidson_data,
    query_precompute,
    get_all_seasons, get_sidebar_stats, _load_truemedia,
    _tm_team, _tm_player, _safe_val, _safe_pct, _safe_num, _tm_pctile,
    _hitter_narrative, _pitcher_narrative,
    _pct_to_float, _clean_pct_cols,
)
from data.stats import (
    compute_batter_stats, compute_pitcher_stats,
    _build_batter_zones,
)
from data.population import (
    compute_batter_stats_pop, compute_pitcher_stats_pop,
    compute_stuff_baselines, build_tunnel_population_pop,
)
