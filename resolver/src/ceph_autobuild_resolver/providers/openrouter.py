"""OpenRouter adapter.

OpenRouter exposes an OpenAI-compatible Chat Completions API. The wire format
is therefore identical to OpenAI's: a ``messages`` array, a ``tools`` array
of ``{type: "function", function: {...}}`` entries, and tool calls returned on
the assistant message with arguments as **JSON-encoded strings** that we have
to ``json.loads`` ourselves.

API reference: https://openrouter.ai/docs/api-reference/chat-completion
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

import httpx

from .base import (
    Message,
    ProviderAdapter,
    ToolCall,
    ToolResult,
    ToolSchema,
    Usage,
)

log = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT = 120.0


class OpenRouterAdapter(ProviderAdapter):
    def __init__(
        self,
        api_key: str,
        model: str,
        url: str = OPENROUTER_URL,
        client: httpx.Client | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._url = url
        # The HTTP client is injectable for tests. In production we own it.
        self._client = client or httpx.Client(timeout=REQUEST_TIMEOUT)
        self._tools_wire: list[dict[str, Any]] = []
        # OpenRouter's unified reasoning param. Shape examples:
        #   {"effort": "medium"}      — let OpenRouter map to the model's budget
        #   {"max_tokens": 4000}      — explicit thinking-token budget
        # When None, the request omits the field entirely (no thinking).
        self._reasoning = reasoning
        # Anthropic prompt caching — mark the static system prompt and initial
        # user message (history[0:2]) with cache_control so turns 2-N don't
        # pay for them again. Only valid for Claude models via OpenRouter.
        self._use_cache = "claude" in model.lower()

    # ------------------------------------------------------------------
    # Tool declaration
    # ------------------------------------------------------------------

    def declare_tools(self, tools: list[ToolSchema]) -> None:
        # OpenAI-style tool list: each tool is wrapped in {type, function}.
        # The ``parameters`` field is a JSON Schema object; we pass through.
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
        body: dict[str, Any] = {
            "model": self._model,
            "messages": _to_wire_messages(history, use_cache=self._use_cache),
        }
        if self._tools_wire:
            body["tools"] = self._tools_wire
        if self._reasoning:
            body["reasoning"] = self._reasoning

        headers: dict[str, str] = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            # OpenRouter requests these for analytics; harmless if absent.
            "HTTP-Referer": "https://github.com/canonical/autobuild-ceph",
            "X-Title": "ceph-autobuild-resolver",
        }
        if self._use_cache:
            headers["anthropic-beta"] = "prompt-caching-2024-07-31"

        log.debug("openrouter request: %d messages", len(body["messages"]))
        resp = self._post_with_retry(headers, body)
        if not resp.is_success:
            log.error("openrouter %d: %s", resp.status_code, resp.text[:2000])
        resp.raise_for_status()
        data = resp.json()
        return _from_wire_response(data)

    def _post_with_retry(
        self,
        headers: dict[str, str],
        body: dict[str, Any],
        max_attempts: int = 5,
        base_delay: float = 5.0,
    ) -> httpx.Response:
        """POST with exponential backoff on transient network/server errors."""
        # 429 and 5xx are server-side transient; ConnectError/TimeoutException
        # are network-level transient. 4xx other than 429 are client errors —
        # retrying won't help.
        retryable_status = {429, 500, 502, 503, 504}
        delay = base_delay
        for attempt in range(1, max_attempts + 1):
            try:
                resp = self._client.post(self._url, headers=headers, json=body)
                if resp.status_code not in retryable_status:
                    return resp
                log.warning(
                    "openrouter %d (attempt %d/%d) — retrying in %.0fs",
                    resp.status_code, attempt, max_attempts, delay,
                )
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                if attempt == max_attempts:
                    raise
                log.warning(
                    "openrouter network error (attempt %d/%d): %s — retrying in %.0fs",
                    attempt, max_attempts, exc, delay,
                )
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        return self._client.post(self._url, headers=headers, json=body)


# ---------------------------------------------------------------------------
# Wire-format translators (free functions for ease of testing)
# ---------------------------------------------------------------------------


def _to_wire_messages(
    history: list[Message], use_cache: bool = False
) -> list[dict[str, Any]]:
    """Translate canonical history into OpenAI-style messages.

    A few non-obvious points:

    * Our internal "model" role becomes "assistant" on the wire.
    * Tool *results* are individual messages with role="tool" and a
      ``tool_call_id`` pointing at the call they answer. If a single canonical
      Message carries multiple ToolResults (parallel execution), it expands
      into multiple wire messages.
    * Tool *calls* live on the assistant message itself, with arguments
      JSON-encoded back into strings (the API requires this even though it
      will parse them server-side).
    * When ``use_cache`` is True, history[0] (system prompt) and history[1]
      (initial user context) are emitted as content-array form with
      ``cache_control`` so Anthropic caches them across turns.
    """
    # history[0] is the system prompt, history[1] is the initial user message.
    # Both are static for the lifetime of a run, making them ideal cache anchors.
    cache_indices = {0, 1} if use_cache else set()

    out: list[dict[str, Any]] = []
    for i, msg in enumerate(history):
        if msg.role == "tool":
            for r in msg.tool_results:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": r.call_id,
                        "name": r.name,
                        "content": json.dumps(r.payload),
                    }
                )
            continue

        wire: dict[str, Any] = {"role": _role_to_wire(msg.role)}
        if msg.text is not None:
            if i in cache_indices:
                wire["content"] = [
                    {
                        "type": "text",
                        "text": msg.text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                wire["content"] = msg.text
        # Anthropic + Gemini reject follow-up turns if the prior assistant
        # message carried thinking blocks and we drop them. Echo verbatim.
        if msg.role == "model" and msg.reasoning_details:
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
            # OpenAI requires assistant messages with tool_calls to also have
            # ``content`` (may be null/empty string).
            wire.setdefault("content", "")
        out.append(wire)
    return out


def _role_to_wire(role: str) -> str:
    return "assistant" if role == "model" else role


def _from_wire_response(data: dict[str, Any]) -> tuple[Message, Usage]:
    """Parse an OpenAI-style response into the canonical reply + usage.

    The interesting part: tool-call ``arguments`` come back as a JSON string
    (e.g. ``'{"path": "foo"}'``). We deserialize once here so every consumer
    downstream sees real Python dicts.
    """
    choice = data["choices"][0]
    msg = choice["message"]

    tool_calls: list[ToolCall] = []
    for tc in msg.get("tool_calls") or []:
        fn = tc["function"]
        raw_args = fn.get("arguments", "")
        try:
            parsed = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            # The model sometimes emits malformed JSON. Surface the raw string
            # so the dispatcher can return a structured error to the model
            # rather than crashing the loop.
            parsed = {"__malformed_arguments__": raw_args}
        tool_calls.append(
            ToolCall(
                # OpenRouter always returns an id, but synthesise one if not.
                id=tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                name=fn["name"],
                args=parsed,
            )
        )

    # OpenRouter normalises thinking output across providers, but inconsistently:
    #   * ``reasoning`` (top-level string)  — set by some providers as a summary.
    #   * ``reasoning_details``             — list of typed blocks; the canonical
    #     form for round-tripping. Shapes seen in the wild:
    #       {"type": "reasoning.text",      "text":    "..."}   (Anthropic)
    #       {"type": "reasoning.summary",   "summary": "..."}   (OpenAI)
    #       {"type": "reasoning.encrypted", "data":    "..."}   (Gemini, opaque)
    # We extract a human-readable summary for display/transcript and keep the
    # full block list verbatim for echoing back on the next turn.
    reasoning_details = list(msg.get("reasoning_details") or [])
    reasoning_text = msg.get("reasoning")
    if not reasoning_text:
        parts = [
            block.get("text") or block.get("summary") or ""
            for block in reasoning_details
        ]
        reasoning_text = "\n".join(p for p in parts if p) or None

    canonical = Message(
        role="model",
        text=msg.get("content") or None,
        reasoning=reasoning_text,
        reasoning_details=reasoning_details,
        tool_calls=tool_calls,
    )

    usage_in = data.get("usage", {}) or {}
    usage = Usage(
        input_tokens=int(usage_in.get("prompt_tokens", 0)),
        output_tokens=int(usage_in.get("completion_tokens", 0)),
    )
    return canonical, usage


# Convenience re-exports so callers can build ToolResults without importing
# from .base.
__all__ = ["OpenRouterAdapter", "ToolResult"]
