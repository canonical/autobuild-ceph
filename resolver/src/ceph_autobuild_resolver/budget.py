"""Budget tracking: token totals, iteration count, and unchanged-iteration streak.

The resolution loop checks ``Budget.has_capacity()`` between turns. When it
runs out, the loop exits and we report failure. Token accounting is
provider-neutral (the adapter normalises to ``Usage``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import Config
from .providers.base import Usage


@dataclass
class Budget:
    cfg: Config
    iterations_used: int = 0
    input_tokens_used: int = 0
    output_tokens_used: int = 0
    unchanged_streak: int = 0
    # True once any run_build has cleared the dpkg-source/patch phase and
    # reached actual compilation (or succeeded outright).
    compilation_reached: bool = False
    _start_time: float = field(default_factory=time.monotonic)

    @classmethod
    def from_config(cls, cfg: Config) -> "Budget":
        return cls(cfg=cfg)

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def record_iteration(self) -> None:
        self.iterations_used += 1

    def record_usage(self, usage: Usage) -> None:
        self.input_tokens_used += usage.input_tokens
        self.output_tokens_used += usage.output_tokens

    def record_compilation_reached(self) -> None:
        """Call when a run_build result shows we cleared the patch/packaging
        phase and actual C++ compilation has started or completed."""
        self.compilation_reached = True

    def record_unchanged_build(self) -> None:
        """Call when run_build failed and no files changed since the previous
        build. Increments the no-progress streak."""
        self.unchanged_streak += 1

    def record_failure(self) -> None:
        """Call when run_build fails after at least one file change. Resets the
        no-progress streak because the model is still making attempts."""
        self.unchanged_streak = 0

    def reset_unchanged_streak(self) -> None:
        self.unchanged_streak = 0

    # ------------------------------------------------------------------
    # Capacity checks
    # ------------------------------------------------------------------

    def _elapsed(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def total_tokens_used(self) -> int:
        return self.input_tokens_used + self.output_tokens_used

    def has_capacity(self) -> bool:
        elapsed = self._elapsed()
        if (
            self.cfg.max_wall_seconds > 0
            and elapsed >= self.cfg.max_wall_seconds
        ):
            return False
        if (
            self.cfg.max_seconds_to_first_build > 0
            and not self.compilation_reached
            and elapsed >= self.cfg.max_seconds_to_first_build
        ):
            return False
        if (
            self.cfg.run_token_budget > 0
            and self.total_tokens_used >= self.cfg.run_token_budget
        ):
            return False
        return (
            self.iterations_used < self.cfg.max_iterations
            and self.unchanged_streak < self.cfg.max_unchanged_iterations
        )

    def reason_for_stop(self) -> str:
        elapsed = self._elapsed()
        if (
            self.cfg.max_wall_seconds > 0
            and elapsed >= self.cfg.max_wall_seconds
        ):
            return "wall_time_exceeded"
        if (
            self.cfg.max_seconds_to_first_build > 0
            and not self.compilation_reached
            and elapsed >= self.cfg.max_seconds_to_first_build
        ):
            return "compilation_not_reached_in_time"
        if (
            self.cfg.run_token_budget > 0
            and self.total_tokens_used >= self.cfg.run_token_budget
        ):
            return "token_budget_exceeded"
        if self.unchanged_streak >= self.cfg.max_unchanged_iterations:
            return "no_progress"
        if self.iterations_used >= self.cfg.max_iterations:
            return "max_iterations"
        return "ok"
