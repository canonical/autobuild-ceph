from __future__ import annotations

import pytest

from ceph_autobuild_resolver import config


def test_load_defaults_to_gemini(monkeypatch):
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gm-test")
    cfg = config.load()
    assert cfg.provider == "gemini"
    assert cfg.api_key == "gm-test"


def test_load_openrouter_explicit(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    cfg = config.load()
    assert cfg.provider == "openrouter"
    assert cfg.api_key == "sk-test"


def test_load_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "claude")
    with pytest.raises(config.ConfigError):
        config.load()


def test_load_requires_api_key(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(config.ConfigError):
        config.load()


def test_int_env_validates(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("MAX_ITERATIONS", "not-an-int")
    with pytest.raises(config.ConfigError):
        config.load()


def _openrouter_env(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")


def test_rejects_negative_reasoning_max_tokens(monkeypatch):
    _openrouter_env(monkeypatch)
    monkeypatch.setenv("REASONING_MAX_TOKENS", "-1")
    with pytest.raises(config.ConfigError, match="REASONING_MAX_TOKENS"):
        config.load()


def test_zero_reasoning_max_tokens_folds_to_none(monkeypatch):
    _openrouter_env(monkeypatch)
    monkeypatch.setenv("REASONING_MAX_TOKENS", "0")
    assert config.load().reasoning_max_tokens is None


def test_rejects_ccache_maxsize_with_shell_metacharacters(monkeypatch):
    _openrouter_env(monkeypatch)
    monkeypatch.setenv("CCACHE_MAXSIZE", "20G; rm -rf /")
    with pytest.raises(config.ConfigError, match="CCACHE_MAXSIZE"):
        config.load()


def test_accepts_valid_ccache_maxsize(monkeypatch):
    _openrouter_env(monkeypatch)
    for value in ("20G", "500M", "5Gi", "1024", "2TiB"):
        monkeypatch.setenv("CCACHE_MAXSIZE", value)
        assert config.load().ccache_max_size == value


def test_api_key_absent_from_repr(monkeypatch):
    _openrouter_env(monkeypatch)
    cfg = config.load()
    assert "sk-test" not in repr(cfg)
    assert cfg.api_key == "sk-test"
