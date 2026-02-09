from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from decision_engine.core.runner_context import RunnerContext


@dataclass(frozen=True)
class BaseState:
    on_1b: bool = False
    on_2b: bool = False
    on_3b: bool = False

    def as_tuple(self) -> Tuple[int, int, int]:
        return (int(bool(self.on_1b)), int(bool(self.on_2b)), int(bool(self.on_3b)))

    @property
    def is_empty(self) -> bool:
        return not (self.on_1b or self.on_2b or self.on_3b)

    @property
    def is_loaded(self) -> bool:
        return self.on_1b and self.on_2b and self.on_3b

    @property
    def risp(self) -> bool:
        return self.on_2b or self.on_3b


@dataclass(frozen=True)
class PitchRecord:
    """Record of a single pitch thrown in the current game."""
    pitch_type: str
    velo: Optional[float] = None
    spin_rate: Optional[float] = None
    ivb: Optional[float] = None
    hb: Optional[float] = None
    result: Optional[str] = None  # "Ball", "Strike", "InPlay", etc.


@dataclass(frozen=True)
class GameState:
    balls: int
    strikes: int
    outs: int = 0
    inning: int = 1
    top_bottom: str = "Top"  # "Top" | "Bot"
    bases: BaseState = BaseState()
    score_our: Optional[int] = None
    score_opp: Optional[int] = None
    runner: RunnerContext = field(default_factory=RunnerContext)
    last_pitch: Optional[str] = None
    # Multi-pitch history: last 3 pitches thrown this AB (most recent last)
    last_pitches: Tuple[str, ...] = ()
    # Full pitch log for this game (for fatigue tracking)
    pitch_log: Tuple[PitchRecord, ...] = ()

    def count(self) -> Tuple[int, int]:
        return (int(self.balls), int(self.strikes))

    def count_str(self) -> str:
        return f"{int(self.balls)}-{int(self.strikes)}"

    @property
    def pitch_count_game(self) -> int:
        """Total pitches thrown this game."""
        return len(self.pitch_log)

    @property
    def leverage_index(self) -> float:
        """Compute a leverage index from game state (0.0 = low, 1.0 = high).

        Factors: score differential, inning, outs, runners on base.
        Modeled after the concept of tBook Leverage Index, simplified for
        real-time use without full win probability tables.
        """
        # Base leverage from inning (later innings = higher leverage)
        inning_factor = min(self.inning / 9.0, 1.5)

        # Score differential: closer games = higher leverage
        if self.score_our is not None and self.score_opp is not None:
            diff = abs(int(self.score_our) - int(self.score_opp))
            if diff == 0:
                score_factor = 1.5   # tie game
            elif diff <= 1:
                score_factor = 1.3   # one-run game
            elif diff <= 2:
                score_factor = 1.0   # two-run game
            elif diff <= 4:
                score_factor = 0.7   # comfortable lead/deficit
            else:
                score_factor = 0.4   # blowout
        else:
            score_factor = 1.0  # unknown scores

        # Runner factor: more runners = more at stake
        runner_count = int(bool(self.bases.on_1b)) + int(bool(self.bases.on_2b)) + int(bool(self.bases.on_3b))
        runner_factor = 1.0 + runner_count * 0.15

        # Outs factor: fewer outs = more at stake
        outs_factor = {0: 1.2, 1: 1.0, 2: 0.9}.get(self.outs, 1.0)

        raw = inning_factor * score_factor * runner_factor * outs_factor
        # Normalize to 0-1 range (typical raw range: 0.2 - 2.7)
        return float(max(0.0, min(1.0, (raw - 0.3) / 2.2)))
