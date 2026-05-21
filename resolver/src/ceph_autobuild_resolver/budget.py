"""Budget tracking: token totals, iteration count, and unchanged-iteration streak.

The resolution loop checks ``Budget.has_capacity()`` between turns. When it
runs out, the loop exits and we report failure. Token accounting is
provider-neutral (the adapter normalises to ``Usage``).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .providers.base import Usage


@dataclass
class Budget:
    cfg: Config
    iterations_used: int = 0
    input_tokens_used: int = 0
    output_tokens_used: int = 0
    unchanged_streak: int = 0

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

    def has_capacity(self) -> bool:
        return (
            self.iterations_used < self.cfg.max_iterations
            and self.input_tokens_used < self.cfg.max_input_tokens
            and self.output_tokens_used < self.cfg.max_output_tokens
            and self.unchanged_streak < self.cfg.max_unchanged_iterations
        )

    def reason_for_stop(self) -> str:
        if self.unchanged_streak >= self.cfg.max_unchanged_iterations:
            return "no_progress"
        if self.iterations_used >= self.cfg.max_iterations:
            return "max_iterations"
        if self.input_tokens_used >= self.cfg.max_input_tokens:
            return "max_input_tokens"
        if self.output_tokens_used >= self.cfg.max_output_tokens:
            return "max_output_tokens"
        return "ok"
