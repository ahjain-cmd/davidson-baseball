from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

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

    def count(self) -> Tuple[int, int]:
        return (int(self.balls), int(self.strikes))

    def count_str(self) -> str:
        return f"{int(self.balls)}-{int(self.strikes)}"
