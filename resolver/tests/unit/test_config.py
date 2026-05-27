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
