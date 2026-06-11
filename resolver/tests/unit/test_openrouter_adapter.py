"""Tests for the OpenRouter adapter — the wire-format translation layer."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from ceph_autobuild_resolver.providers.base import (
    Message,
    ToolCall,
    ToolResult,
    ToolSchema,
)
from ceph_autobuild_resolver.providers.openrouter import (
    OpenRouterAdapter,
    _from_wire_response,
    _to_wire_messages,
)


def test_to_wire_messages_translates_roles_and_tool_calls():
    history = [
        Message(role="system", text="be helpful"),
        Message(role="user", text="hi"),
        Message(
            role="model",
            text=None,
            tool_calls=[
                ToolCall(id="call_1", name="read_files", args={"paths": ["a"]})
            ],
        ),
        Message(
            role="tool",
            tool_results=[
                ToolResult(
                    call_id="call_1",
                    name="read_files",
                    payload={"files": [{"path": "a", "content": "x"}]},
                )
            ],
        ),
    ]

    wire = _to_wire_messages(history)

    assert wire[0] == {"role": "system", "content": "be helpful"}
    assert wire[1] == {"role": "user", "content": "hi"}
    # Model role becomes "assistant" on the wire.
    assert wire[2]["role"] == "assistant"
    # Arguments are JSON-encoded into a string for the OpenAI-style API.
    fn_args = wire[2]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(fn_args) == {"paths": ["a"]}
    # Tool result is its own message with tool_call_id matching the call.
    assert wire[3]["role"] == "tool"
    assert wire[3]["tool_call_id"] == "call_1"
    assert json.loads(wire[3]["content"]) == {
        "files": [{"path": "a", "content": "x"}]
    }


def test_from_wire_response_parses_tool_call_arguments():
    data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_42",
                            "type": "function",
                            "function": {
                                "name": "grep",
                                "arguments": '{"pattern": "TODO"}',
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 30},
    }

    msg, usage = _from_wire_response(data)

    assert msg.role == "model"
    assert len(msg.tool_calls) == 1
    # The whole point of the adapter: caller sees a real dict, not a string.
    assert msg.tool_calls[0].args == {"pattern": "TODO"}
    assert usage.input_tokens == 100
    assert usage.output_tokens == 30


def test_from_wire_response_handles_malformed_arguments():
    data = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "grep",
                                "arguments": "{not json",
                            },
                        }
                    ]
                }
            }
        ],
        "usage": {},
    }
    msg, _ = _from_wire_response(data)
    assert "__malformed_arguments__" in msg.tool_calls[0].args


def test_chat_round_trips_with_mock_transport():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers["Authorization"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                            "tool_calls": [],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    adapter = OpenRouterAdapter(api_key="sk-test", model="m", http_client=client)
    adapter.declare_tools(
        [
            ToolSchema(
                name="grep",
                description="d",
                parameters={"type": "object", "properties": {}},
            )
        ]
    )

    reply, usage = adapter.chat([Message(role="user", text="hello")])

    assert reply.text == "ok"
    assert usage.input_tokens == 5
    # Tools were forwarded.
    assert captured["body"]["tools"][0]["function"]["name"] == "grep"
    assert captured["auth"] == "Bearer sk-test"


def test_from_wire_response_extracts_reasoning_summary():
    data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "ok",
                    "reasoning": "I should grep for it.",
                    "reasoning_details": [
                        {"type": "reasoning.text", "text": "I should grep for it."}
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    msg, _ = _from_wire_response(data)
    assert msg.reasoning == "I should grep for it."
    assert msg.reasoning_details == [
        {"type": "reasoning.text", "text": "I should grep for it."}
    ]


def test_from_wire_response_synthesises_summary_from_details():
    """When the top-level ``reasoning`` field is absent (Gemini, OpenAI), fall
    back to concatenating per-block text/summary fields."""
    data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_details": [
                        {"type": "reasoning.summary", "summary": "step one"},
                        {"type": "reasoning.summary", "summary": "step two"},
                    ],
                }
            }
        ],
        "usage": {},
    }
    msg, _ = _from_wire_response(data)
    assert msg.reasoning == "step one\nstep two"


def test_to_wire_messages_echoes_reasoning_details_back():
    """Anthropic + Gemini reject the next turn if prior thinking blocks are
    dropped — we must round-trip ``reasoning_details`` verbatim."""
    history = [
        Message(
            role="model",
            text=None,
            reasoning_details=[{"type": "reasoning.encrypted", "data": "OPAQUE"}],
            tool_calls=[ToolCall(id="c1", name="grep", args={"pattern": "x"})],
        )
    ]
    wire = _to_wire_messages(history)
    assert wire[0]["reasoning_details"] == [
        {"type": "reasoning.encrypted", "data": "OPAQUE"}
    ]


def test_chat_forwards_reasoning_param():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "ok"}}
                ],
                "usage": {},
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    adapter = OpenRouterAdapter(
        api_key="sk-test",
        model="m",
        http_client=client,
        reasoning={"max_tokens": 4000},
    )
    adapter.chat([Message(role="user", text="hi")])
    assert captured["body"]["reasoning"] == {"max_tokens": 4000}


def test_chat_omits_reasoning_when_disabled():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "ok"}}
                ],
                "usage": {},
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    adapter = OpenRouterAdapter(api_key="sk-test", model="m", http_client=client)
    adapter.chat([Message(role="user", text="hi")])
    assert "reasoning" not in captured["body"]


def test_assistant_with_tool_calls_has_content_field():
    """OpenAI-compatible APIs require ``content`` even when only tool_calls
    are present; we default to empty string."""
    history = [
        Message(
            role="model",
            text=None,
            tool_calls=[ToolCall(id="c1", name="grep", args={"pattern": "x"})],
        )
    ]
    wire = _to_wire_messages(history)
    assert "content" in wire[0]


# ---------------------------------------------------------------------------
# Prompt caching
# ---------------------------------------------------------------------------


def test_to_wire_messages_cache_control_on_first_two():
    """When use_cache=True, indices 0 and 1 get content-array form with cache_control."""
    history = [
        Message(role="system", text="be helpful"),
        Message(role="user", text="initial context"),
        Message(role="user", text="follow up"),
    ]
    wire = _to_wire_messages(history, use_cache=True)

    # Index 0: system — should be content-array with cache_control.
    assert isinstance(wire[0]["content"], list)
    assert wire[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert wire[0]["content"][0]["text"] == "be helpful"

    # Index 1: first user — should also have cache_control.
    assert isinstance(wire[1]["content"], list)
    assert wire[1]["content"][0]["cache_control"] == {"type": "ephemeral"}

    # Index 2: second user — plain string, no cache_control.
    assert wire[2]["content"] == "follow up"


def test_to_wire_messages_no_cache_control_by_default():
    """Without use_cache, content stays as a plain string (backwards compat)."""
    history = [
        Message(role="system", text="be helpful"),
        Message(role="user", text="hi"),
    ]
    wire = _to_wire_messages(history)
    assert wire[0]["content"] == "be helpful"
    assert wire[1]["content"] == "hi"


def test_chat_sends_anthropic_beta_header_for_claude_model():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    adapter = OpenRouterAdapter(
        api_key="sk-test",
        model="anthropic/claude-sonnet-4-5",
        http_client=client,
    )
    adapter.chat([Message(role="user", text="hi")])
    assert captured["headers"].get("anthropic-beta") == "prompt-caching-2024-07-31"


def test_chat_no_anthropic_beta_header_for_non_claude_model():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {},
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    adapter = OpenRouterAdapter(
        api_key="sk-test",
        model="google/gemini-2.0-flash",
        http_client=client,
    )
    adapter.chat([Message(role="user", text="hi")])
    assert "anthropic-beta" not in captured["headers"]


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

_OK_RESPONSE = {
    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
}


def test_retry_on_connect_error_then_success():
    """A ConnectError on the first attempt is retried and succeeds."""
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ConnectError("name resolution failed")
        return httpx.Response(200, json=_OK_RESPONSE)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    adapter = OpenRouterAdapter(api_key="sk-test", model="m", http_client=client)

    with patch("time.sleep"):
        reply, usage = adapter.chat([Message(role="user", text="hi")])

    assert reply.text == "ok"
    assert attempts["n"] == 2


def test_retry_on_503_then_success():
    """A 503 on the first attempt is retried and succeeds."""
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(503, text="service unavailable")
        return httpx.Response(200, json=_OK_RESPONSE)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    adapter = OpenRouterAdapter(api_key="sk-test", model="m", http_client=client)

    with patch("time.sleep"):
        reply, _ = adapter.chat([Message(role="user", text="hi")])

    assert reply.text == "ok"
    assert attempts["n"] == 2


def test_connect_error_exhausts_retries():
    """Persistent ConnectError re-raises (wrapped by the SDK) after max retries."""
    import openai

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("always failing")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    adapter = OpenRouterAdapter(api_key="sk-test", model="m", http_client=client)

    with patch("time.sleep"), pytest.raises(openai.APIConnectionError):
        adapter.chat([Message(role="user", text="hi")])


# ---------------------------------------------------------------------------
# Error handling: context overflow, empty choices, truncation
# ---------------------------------------------------------------------------

import pytest
from openai import BadRequestError

from ceph_autobuild_resolver.providers.base import ContextOverflowError


def _adapter_with(handler) -> OpenRouterAdapter:
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    return OpenRouterAdapter(api_key="sk-test", model="m", http_client=client)


def test_chat_maps_context_limit_400_to_context_overflow():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {
                "message": (
                    "This endpoint's maximum context length is 200000 tokens. "
                    "However, you requested 215000 tokens."
                ),
                "type": "invalid_request_error",
            }},
        )

    with pytest.raises(ContextOverflowError):
        _adapter_with(handler).chat([Message(role="user", text="hi")])


def test_chat_reraises_unrelated_400():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "Invalid parameter: tools"}},
        )

    with pytest.raises(BadRequestError):
        _adapter_with(handler).chat([Message(role="user", text="hi")])


def test_empty_choices_returns_blocked_turn():
    """OpenRouter can return HTTP 200 with empty choices and an in-body
    provider error; that must come back as a recoverable [BLOCKED] turn."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [],
                "error": {"message": "Provider returned error", "code": 502},
                "usage": {"prompt_tokens": 7, "completion_tokens": 0},
            },
        )

    reply, usage = _adapter_with(handler).chat([Message(role="user", text="hi")])
    assert "[BLOCKED" in reply.text
    assert usage.input_tokens == 7


def test_finish_reason_length_marks_truncated_reply():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": "partial answer"},
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 9},
            },
        )

    reply, _usage = _adapter_with(handler).chat([Message(role="user", text="hi")])
    assert "partial answer" in reply.text
    assert "[TRUNCATED" in reply.text
