"""Decision engine package (in-game pitch/location recommendations).

This package is intentionally lightweight and primarily wraps the existing
scoring infrastructure in `_pages/scouting.py` with:
  - Bayesian-style shrinkage for small samples
  - Count- and base-state-aware adjustments
  - A Streamlit UI panel for in-game use
"""

from .core.state import GameState, BaseState
from .core.runner_context import RunnerContext
from .core.matchup import score_pitcher_vs_hitter_shrunk
from .recommenders.pitch_call import recommend_pitch_call

__all__ = [
    "BaseState",
    "GameState",
    "RunnerContext",
    "score_pitcher_vs_hitter_shrunk",
    "recommend_pitch_call",
]
