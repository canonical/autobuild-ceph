"""Configuration loaded from environment variables.

Read once at startup and passed (frozen) to every other module so that no code
ever reaches into ``os.environ`` ad-hoc.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

ProviderName = Literal["openrouter", "gemini"]


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Config:
    # Provider selection
    provider: ProviderName
    model_name: str
    api_key: str

    # Budget caps (per-failure)
    max_input_tokens: int
    max_output_tokens: int
    max_iterations: int
    max_unchanged_iterations: int

    # Cross-matrix (per CI run) cap. Tracked but enforced by the caller.
    run_token_budget: int

    # Build matrix knobs (mirrored from build.sh)
    ubuntu_branch: str
    debian_ref: str
    launchpad_owner: str
    ceph_version: str

    # Wall-clock time limits (seconds). 0 = disabled.
    # max_wall_seconds: hard cap on total loop duration.
    # max_seconds_to_first_build: stop if run_build hasn't been called within
    #   this many seconds of loop start — catches a model stuck in pure diagnosis.
    max_wall_seconds: int = 0
    max_seconds_to_first_build: int = 0

    # Reasoning / thinking. At most one is honoured; max_tokens wins.
    # None on both = thinking disabled (no `reasoning` field sent).
    reasoning_effort: str | None = None
    reasoning_max_tokens: int | None = None

    # Working paths inside the container
    container_workdir: str = "/root/ceph"
    container_log_dir: str = "/root/build-logs"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def load() -> Config:
    """Build a Config from the current environment."""
    provider = os.environ.get("MODEL_PROVIDER", "gemini").lower()
    if provider not in ("openrouter", "gemini"):
        raise ConfigError(
            f"MODEL_PROVIDER must be 'openrouter' or 'gemini', got {provider!r}"
        )

    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        default_model = "anthropic/claude-sonnet-4-5"
    else:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        default_model = "gemini-3.1-pro-preview"

    if not api_key:
        raise ConfigError(
            f"Missing API key for provider {provider!r} "
            f"(set OPENROUTER_API_KEY or GEMINI_API_KEY)"
        )

    effort = os.environ.get("REASONING_EFFORT") or None
    if effort is not None and effort not in ("low", "medium", "high"):
        raise ConfigError(
            f"REASONING_EFFORT must be one of low/medium/high, got {effort!r}"
        )
    max_thinking = _env_int("REASONING_MAX_TOKENS", 0) or None

    return Config(
        provider=provider,  # type: ignore[arg-type]
        model_name=os.environ.get("MODEL_NAME", default_model),
        api_key=api_key,
        max_input_tokens=_env_int("MAX_INPUT_TOKENS", 8_000_000),
        max_output_tokens=_env_int("MAX_OUTPUT_TOKENS", 8_000_000),
        max_iterations=_env_int("MAX_ITERATIONS", 20),
        max_unchanged_iterations=_env_int("MAX_UNCHANGED_ITERATIONS", 3),
        run_token_budget=_env_int("RUN_TOKEN_BUDGET", 16_000_000),
        max_wall_seconds=_env_int("MAX_WALL_SECONDS", 0),
        max_seconds_to_first_build=_env_int("MAX_SECONDS_TO_FIRST_BUILD", 0),
        ubuntu_branch=os.environ.get("UBUNTU_BRANCH", "ubuntu/resolute"),
        debian_ref=os.environ.get("DEBIAN_REF", "origin/ubuntu/latest"),
        launchpad_owner=os.environ.get("LAUNCHPAD_OWNER", "lmlogiudice"),
        ceph_version=os.environ.get("CEPH_VERSION", "20.2.0"),
        reasoning_effort=effort,
        reasoning_max_tokens=max_thinking,
    )
