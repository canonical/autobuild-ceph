"""Pick the right adapter based on Config."""

from __future__ import annotations

from ..config import Config
from .base import ProviderAdapter
from .gemini import GeminiAdapter
from .openrouter import OpenRouterAdapter


def build(cfg: Config) -> ProviderAdapter:
    if cfg.provider == "openrouter":
        # max_tokens wins over effort (more deterministic budget control).
        reasoning: dict | None = None
        if cfg.reasoning_max_tokens:
            reasoning = {"max_tokens": cfg.reasoning_max_tokens}
        elif cfg.reasoning_effort:
            reasoning = {"effort": cfg.reasoning_effort}
        return OpenRouterAdapter(
            api_key=cfg.api_key,
            model=cfg.model_name,
            reasoning=reasoning,
        )
    if cfg.provider == "gemini":
        # Thinking is always enabled. REASONING_MAX_TOKENS caps the per-call
        # thinking token budget; if unset, the model decides its own budget.
        # REASONING_EFFORT is not supported by Gemini (OpenRouter-only).
        return GeminiAdapter(
            api_key=cfg.api_key,
            model=cfg.model_name,
            thinking_budget=cfg.reasoning_max_tokens or None,
        )
    raise ValueError(f"unknown provider: {cfg.provider!r}")
