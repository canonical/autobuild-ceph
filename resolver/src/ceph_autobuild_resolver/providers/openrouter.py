"""OpenRouter adapter using the openai SDK.

OpenRouter exposes an OpenAI-compatible Chat Completions endpoint, so the
``openai`` SDK works with only a ``base_url`` override.  The SDK handles
auth headers, connection pooling, and retry-with-backoff; we focus on
translating between the canonical message types and the wire format.

Non-standard OpenRouter extensions surfaced here:
* ``reasoning`` body field — enables chain-of-thought for supported models.
* ``reasoning_details`` / ``reasoning`` message fields — echoed back verbatim
  so Anthropic/Gemini models don't reject the next turn.
* ``cache_control`` content blocks + ``anthropic-beta`` header — Anthropic
  prompt caching via OpenRouter.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import httpx
from openai import BadRequestError, OpenAI

from .base import ROLE_MODEL
from .base import (
    ContextOverflowError,
    Message,
    ProviderAdapter,
    ToolCall,
    ToolResult,
    ToolSchema,
    Usage,
)

log = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Phrasings that identify a 400 as "request exceeded the context window".
# OpenRouter itself says "This endpoint's maximum context length is N
# tokens..."; Anthropic-via-OpenRouter says "prompt is too long: X tokens
# > Y maximum". Kept narrow for the same reason as the Gemini markers.
_CONTEXT_LIMIT_MARKERS = (
    "maximum context length",
    "context length exceeded",
    "prompt is too long",
    "input token count",
)


def _is_context_limit_error(exc: BadRequestError) -> bool:
    msg = (str(exc) or "").lower()
    return any(marker in msg for marker in _CONTEXT_LIMIT_MARKERS)


class OpenRouterAdapter(ProviderAdapter):
    def __init__(
        self,
        api_key: str,
        model: str,
        reasoning: dict[str, Any] | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._model = model
        self._reasoning = reasoning
        self._use_cache = "claude" in model.lower()
        self._tools_wire: list[dict[str, Any]] = []

        default_headers: dict[str, str] = {
            "HTTP-Referer": "https://github.com/canonical/autobuild-ceph",
            "X-Title": "ceph-autobuild-resolver",
        }
        if self._use_cache:
            default_headers["anthropic-beta"] = "prompt-caching-2024-07-31"

        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            max_retries=5,
            default_headers=default_headers,
            http_client=http_client,
        )

    # ------------------------------------------------------------------
    # Tool declaration
    # ------------------------------------------------------------------

    def declare_tools(self, tools: list[ToolSchema]) -> None:
        self._tools_wire = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    # ------------------------------------------------------------------
    # Round-trip
    # ------------------------------------------------------------------

    def chat(self, history: list[Message]) -> tuple[Message, Usage]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": _to_wire_messages(history, use_cache=self._use_cache),
        }
        if self._tools_wire:
            kwargs["tools"] = self._tools_wire
        if self._reasoning:
            kwargs["extra_body"] = {"reasoning": self._reasoning}

        log.debug("openrouter request: %d messages", len(kwargs["messages"]))
        try:
            response = self._client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            # Per the base.py contract every adapter translates its own
            # "request too large" failure so the loop can stop cleanly with
            # a recorded outcome. Other 400s are real errors — re-raise.
            if _is_context_limit_error(exc):
                raise ContextOverflowError(str(exc)) from exc
            raise
        return _from_sdk_response(response)


# ---------------------------------------------------------------------------
# Wire-format translators (module-level for unit-testability)
# ---------------------------------------------------------------------------


def _to_wire_messages(
    history: list[Message], use_cache: bool = False
) -> list[dict[str, Any]]:
    """Translate canonical history into OpenAI-style message dicts.

    * Internal ``model`` role → ``assistant`` on the wire.
    * Tool results expand from a single canonical Message into one wire message
      per result (OpenAI requires role=``tool`` per call-id).
    * Tool call arguments are JSON-encoded back to strings (API requirement).
    * When ``use_cache`` is True, history[0] and history[1] gain
      ``cache_control`` so Anthropic caches them across turns.
    """
    # Assumes history[0]=system prompt and history[1]=initial user context message,
    # both static for the lifetime of a run and therefore ideal Anthropic cache anchors.
    cache_indices = {0, 1} if use_cache else set()
    out: list[dict[str, Any]] = []

    for i, msg in enumerate(history):
        if msg.role == "tool":
            for r in msg.tool_results:
                out.append({
                    "role": "tool",
                    "tool_call_id": r.call_id,
                    "name": r.name,
                    "content": json.dumps(r.payload),
                })
            continue

        wire: dict[str, Any] = {"role": _role_to_wire(msg.role)}

        if msg.text is not None:
            if i in cache_indices:
                wire["content"] = [{
                    "type": "text",
                    "text": msg.text,
                    "cache_control": {"type": "ephemeral"},
                }]
            else:
                wire["content"] = msg.text

        if msg.role == ROLE_MODEL and msg.reasoning_details:
            wire["reasoning_details"] = msg.reasoning_details

        if msg.tool_calls:
            wire["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.args),
                    },
                }
                for tc in msg.tool_calls
            ]
            wire.setdefault("content", None)

        out.append(wire)

    return out


def _role_to_wire(role: str) -> str:
    return "assistant" if role == ROLE_MODEL else role


def _from_sdk_response(response: Any) -> tuple[Message, Usage]:
    """Parse an openai SDK ChatCompletion into canonical reply + usage."""
    if not response.choices:
        # OpenRouter can return HTTP 200 with empty choices and an in-body
        # provider error. Surface as a recoverable [BLOCKED] turn (mirrors
        # the Gemini adapter) instead of crashing on choices[0].
        err = (getattr(response, "model_extra", None) or {}).get("error")
        log.error("openrouter returned no choices (error=%s)", err)
        usage = Usage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )
        return Message(
            role=ROLE_MODEL,
            text=f"[BLOCKED: provider returned no choices; error={err}]",
        ), usage

    choice = response.choices[0]
    msg = choice.message

    tool_calls: list[ToolCall] = []
    for tc in msg.tool_calls or []:
        raw_args = tc.function.arguments or ""
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args = {"__malformed_arguments__": raw_args}
        tool_calls.append(ToolCall(
            id=tc.id or f"call_{uuid.uuid4().hex[:12]}",
            name=tc.function.name,
            args=args,
        ))

    # reasoning and reasoning_details are OpenRouter extensions that arrive
    # in the message's extra fields (Pydantic v2 extra='allow').
    extra = msg.model_extra or {}
    reasoning_details = list(extra.get("reasoning_details") or [])
    reasoning_text = extra.get("reasoning")
    if not reasoning_text and reasoning_details:
        parts = [b.get("text") or b.get("summary") or "" for b in reasoning_details]
        reasoning_text = "\n".join(p for p in parts if p) or None

    usage = Usage(
        input_tokens=response.usage.prompt_tokens if response.usage else 0,
        output_tokens=response.usage.completion_tokens if response.usage else 0,
    )

    text = msg.content or None
    if getattr(choice, "finish_reason", None) == "length":
        # A length-truncated reply is otherwise indistinguishable from a
        # complete one; mark it so the loop/transcript see why the turn
        # was cut short.
        log.warning("openrouter reply truncated: finish_reason=length")
        if not tool_calls:
            text = f"{text or ''}\n[TRUNCATED: finish_reason=length — reply hit the output-token limit]".strip()

    return Message(
        role=ROLE_MODEL,
        text=text,
        reasoning=reasoning_text,
        reasoning_details=reasoning_details,
        tool_calls=tool_calls,
    ), usage


# Keep _from_wire_response for direct unit tests that build mock response dicts.
def _from_wire_response(data: dict[str, Any]) -> tuple[Message, Usage]:
    """Parse a raw OpenAI-format response dict — used by unit tests."""
    choice = data["choices"][0]
    msg = choice["message"]

    tool_calls: list[ToolCall] = []
    for tc in msg.get("tool_calls") or []:
        fn = tc["function"]
        raw_args = fn.get("arguments", "")
        try:
            parsed = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            parsed = {"__malformed_arguments__": raw_args}
        tool_calls.append(ToolCall(
            id=tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
            name=fn["name"],
            args=parsed,
        ))

    reasoning_details = list(msg.get("reasoning_details") or [])
    reasoning_text = msg.get("reasoning")
    if not reasoning_text and reasoning_details:
        parts = [b.get("text") or b.get("summary") or "" for b in reasoning_details]
        reasoning_text = "\n".join(p for p in parts if p) or None

    usage_in = data.get("usage", {}) or {}
    return Message(
        role=ROLE_MODEL,
        text=msg.get("content") or None,
        reasoning=reasoning_text,
        reasoning_details=reasoning_details,
        tool_calls=tool_calls,
    ), Usage(
        input_tokens=int(usage_in.get("prompt_tokens", 0)),
        output_tokens=int(usage_in.get("completion_tokens", 0)),
    )
