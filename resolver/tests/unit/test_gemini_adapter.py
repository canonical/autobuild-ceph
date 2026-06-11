"""Gemini adapter: context-overflow classification.

Verifies the adapter maps a 400 *token-limit* rejection to the provider-neutral
ContextOverflowError, while re-raising every other 400 unchanged so genuine
errors are not silently swallowed.
"""

from __future__ import annotations

import pytest
from google.genai import errors as genai_errors

from ceph_autobuild_resolver.providers.base import ContextOverflowError, Message
from ceph_autobuild_resolver.providers.gemini import (
    GeminiAdapter,
    _is_token_limit_error,
)


def _client_error(code: int, message: str) -> genai_errors.ClientError:
    return genai_errors.ClientError(
        code,
        {"error": {"code": code, "message": message, "status": "INVALID_ARGUMENT"}},
        None,
    )


_TOKEN_LIMIT_MSG = (
    "The input token count exceeds the maximum number of tokens allowed 1048576."
)
_OTHER_400_MSG = "Invalid JSON payload received. Unknown name 'foo'."


def test_is_token_limit_error_discriminates():
    assert _is_token_limit_error(_client_error(400, _TOKEN_LIMIT_MSG)) is True
    assert _is_token_limit_error(_client_error(400, _OTHER_400_MSG)) is False


def test_chat_maps_token_limit_400_to_context_overflow(monkeypatch):
    adapter = GeminiAdapter(api_key="x", model="gemini-test")

    def boom(*args, **kwargs):
        raise _client_error(400, _TOKEN_LIMIT_MSG)

    monkeypatch.setattr(adapter._client.models, "generate_content", boom)
    with pytest.raises(ContextOverflowError):
        adapter.chat([Message(role="user", text="hi")])


def test_chat_reraises_unrelated_400(monkeypatch):
    adapter = GeminiAdapter(api_key="x", model="gemini-test")

    def boom(*args, **kwargs):
        raise _client_error(400, _OTHER_400_MSG)

    monkeypatch.setattr(adapter._client.models, "generate_content", boom)
    with pytest.raises(genai_errors.ClientError):
        adapter.chat([Message(role="user", text="hi")])


def test_client_configured_with_retries():
    """google-genai defaults to zero retries; the adapter must opt in so a
    transient 429/5xx cannot destroy a multi-hour run."""
    adapter = GeminiAdapter(api_key="x", model="gemini-test")
    retry_options = adapter._client._api_client._http_options.retry_options
    assert retry_options is not None
