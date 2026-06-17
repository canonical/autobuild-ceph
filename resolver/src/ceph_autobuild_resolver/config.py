"""Configuration loaded from environment variables.

Read once at startup and passed (frozen) to every other module so that no code
ever reaches into ``os.environ`` ad-hoc.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Literal

ProviderName = Literal["openrouter", "gemini"]

# Per-exec timeouts (seconds), enforced in-container via coreutils ``timeout``
# (see LXDManager.exec). pylxd's execute() otherwise blocks forever on a hung
# command, and the loop's wall-clock budget is only checked between turns.
#
# TOOL_EXEC_TIMEOUT_SECONDS: short cap for model tool calls (reads, greps,
# patch dry-runs) -- none of these should take more than a couple of minutes.
# BUILD_STEP_TIMEOUT_SECONDS: generous cap for a single build/prep step;
# a full Ceph compile fits comfortably, a wedged debuild does not run forever.
TOOL_EXEC_TIMEOUT_SECONDS = 120
BUILD_STEP_TIMEOUT_SECONDS = 6 * 3600


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Config:
    # Provider selection
    provider: ProviderName
    model_name: str
    # repr=False keeps the key out of the dataclass's auto-generated __repr__,
    # so it never leaks into a debug print(cfg) or an unhandled-exception
    # traceback that renders the Config.
    api_key: str = field(repr=False)

    # Budget caps (per-failure)
    max_iterations: int
    max_unchanged_iterations: int

    # Per-process token cap, enforced by Budget between iterations. Each
    # matrix job is a separate process with its own budget; nothing enforces
    # a cross-matrix total.
    run_token_budget: int

    # Build matrix knobs (mirrored from build.sh)
    ubuntu_branch: str
    debian_ref: str
    launchpad_owner: str
    ceph_version: str

    # Wall-clock time limits (seconds). 0 = disabled.
    # max_wall_seconds: hard cap on total loop duration.
    # max_seconds_to_first_build: stop if the build hasn't cleared the patch/packaging
    #   phase and reached actual compilation within this many seconds — catches a
    #   model stuck in pure diagnosis or a patch that never applies.
    max_wall_seconds: int = 0
    max_seconds_to_first_build: int = 0

    # Reasoning / thinking. At most one is honoured; max_tokens wins.
    # None on both = thinking disabled (no `reasoning` field sent).
    reasoning_effort: str | None = None
    reasoning_max_tokens: int | None = None

    # ccache: if set, this host directory is bind-mounted into containers as
    # /root/ccache and used as CCACHE_DIR during builds.  None = ccache disabled.
    ccache_host_dir: str | None = None
    # ccache cache-size cap passed to the build as CCACHE_MAXSIZE.  Accepts
    # ccache's size syntax (e.g. "20G", "5G", "500M").  Ignored when ccache
    # is disabled.
    ccache_max_size: str = "20G"

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
    if max_thinking is not None and max_thinking < 0:
        # `or None` only folds 0 away; a negative is truthy and would reach
        # Gemini's ThinkingConfig(thinking_budget=-1), which 400s.
        raise ConfigError(
            f"REASONING_MAX_TOKENS must be a non-negative integer, got {max_thinking}"
        )

    # CCACHE_MAXSIZE is interpolated into the build's shell command line, so
    # restrict it to a ccache size token (digits + optional unit) at load time
    # rather than trusting an arbitrary env value to be shell-safe.
    ccache_max_size = os.environ.get("CCACHE_MAXSIZE") or "20G"
    if not re.fullmatch(r"[0-9]+[KkMmGgTtPp]?i?[Bb]?", ccache_max_size):
        raise ConfigError(
            f"CCACHE_MAXSIZE must be a ccache size string (e.g. '20G', '500M'), "
            f"got {ccache_max_size!r}"
        )

    return Config(
        provider=provider,  # type: ignore[arg-type]
        model_name=os.environ.get("MODEL_NAME", default_model),
        api_key=api_key,
        max_iterations=_env_int("MAX_ITERATIONS", 20),
        max_unchanged_iterations=_env_int("MAX_UNCHANGED_ITERATIONS", 3),
        run_token_budget=_env_int("RUN_TOKEN_BUDGET", 32_000_000),
        max_wall_seconds=_env_int("MAX_WALL_SECONDS", 0),
        max_seconds_to_first_build=_env_int("MAX_SECONDS_TO_FIRST_BUILD", 0),
        ubuntu_branch=os.environ.get("UBUNTU_BRANCH", "ubuntu/resolute"),
        debian_ref=os.environ.get("DEBIAN_REF", "origin/ubuntu/latest"),
        launchpad_owner=os.environ.get("LAUNCHPAD_OWNER", "lmlogiudice"),
        ceph_version=os.environ.get("CEPH_VERSION", "20.2.0"),
        reasoning_effort=effort,
        reasoning_max_tokens=max_thinking,
        ccache_host_dir=os.environ.get("CCACHE_HOST_DIR") or None,
        ccache_max_size=ccache_max_size,
    )
