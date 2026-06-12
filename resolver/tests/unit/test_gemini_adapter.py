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


# ---------------------------------------------------------------------------
# Response parsing robustness
# ---------------------------------------------------------------------------

from types import SimpleNamespace as _NS

from ceph_autobuild_resolver.providers.gemini import _from_response


def _meta(prompt=100, cand=5, thoughts=0):
    return _NS(
        candidates_token_count=cand,
        prompt_token_count=prompt,
        thoughts_token_count=thoughts,
    )


def test_from_response_handles_no_candidates():
    """A prompt-blocked response has no candidates at all; it must come back
    as a recoverable [BLOCKED] turn, not an IndexError."""
    resp = _NS(
        candidates=[],
        prompt_feedback="BLOCK_REASON_SAFETY",
        usage_metadata=_meta(prompt=1234, cand=0),
    )
    msg, usage = _from_response(resp)
    assert "[BLOCKED" in msg.text
    assert "SAFETY" in msg.text
    assert usage.input_tokens == 1234


def test_blocked_finish_reason_still_counts_usage():
    cand = _NS(finish_reason=_NS(name="SAFETY"), content=None)
    resp = _NS(candidates=[cand], usage_metadata=_meta(prompt=50, cand=0))
    msg, usage = _from_response(resp)
    assert "[BLOCKED" in msg.text
    assert usage.input_tokens == 50


def test_text_part_thought_signature_preserved():
    part = _NS(function_call=None, text="hello", thought=False, thought_signature=b"sig")
    cand = _NS(finish_reason=_NS(name="STOP"), content=_NS(parts=[part]))
    resp = _NS(candidates=[cand], usage_metadata=_meta())
    msg, _usage = _from_response(resp)
    assert msg.text == "hello"
    assert msg.thought_signature == b"sig"


def test_overflow_marker_ignores_unrelated_maximum_phrasings():
    exc = _client_error(
        400, "Value exceeds the maximum allowed range for parameter temperature."
    )
    assert _is_token_limit_error(exc) is False


def test_to_content_echoes_thought_signature_on_tool_only_turn():
    """The dominant agentic turn is tool calls with no prose (text=None) plus a
    thought_signature. The signature must still be echoed on a carrying Part, or
    Gemini rejects the next turn for a thinking model."""
    from ceph_autobuild_resolver.providers.base import Message, ToolCall
    from ceph_autobuild_resolver.providers.gemini import _to_content

    msg = Message(
        role="model",
        text=None,
        thought_signature=b"thought_sig",
        tool_calls=[ToolCall(id="c1", name="run_build", args={})],
    )
    content = _to_content(msg)
    sigs = [getattr(p, "thought_signature", None) for p in content.parts]
    assert b"thought_sig" in sigs
