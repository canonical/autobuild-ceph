"""Gemini adapter using the google-genai SDK.

Wire-format differences from OpenRouter/OpenAI handled by the SDK:
* System prompt → ``GenerateContentConfig.system_instruction``.
* Tool declarations → ``types.Tool(function_declarations=[...])``.
* Tool call args arrive as parsed dicts (no json.loads needed).
* Tool results attach as ``functionResponse`` parts on a user turn.
* Token counts come from ``response.usage_metadata``.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from .base import (
    ROLE_MODEL,
    ContextOverflowError,
    Message,
    ProviderAdapter,
    ToolCall,
    ToolSchema,
    Usage,
)

log = logging.getLogger(__name__)

# Substrings that mark a 400 as a per-request token-limit rejection (as opposed
# to other INVALID_ARGUMENT causes like a malformed schema). Matched only as a
# secondary gate behind the HTTP 400 code check, never on its own.
# Deliberately narrow: only token-count phrasings. A broad substring like
# "exceeds the maximum" also matches unrelated 400s (parameter range
# messages), which would turn a real bug into a silent clean stop.
_TOKEN_LIMIT_MARKERS = (
    "exceeds the maximum number of tokens",
    "input token count",
)


def _is_token_limit_error(exc: genai_errors.ClientError) -> bool:
    msg = (getattr(exc, "message", None) or str(exc) or "").lower()
    return any(marker in msg for marker in _TOKEN_LIMIT_MARKERS)

# finish_reason values that indicate the model stopped normally.
_OK_FINISH_REASONS = frozenset({"STOP", "MAX_TOKENS", "FINISH_REASON_UNSPECIFIED"})
_SCHEMA_UNSUPPORTED = frozenset({
    "$schema", "$id", "$ref",
    "additionalProperties", "unevaluatedProperties",
    "allOf", "not", "if", "then", "else",
})


class GeminiAdapter(ProviderAdapter):
    def __init__(self, api_key: str, model: str, thinking_budget: int | None = None) -> None:
        self._model = model
        # google-genai performs NO retries by default (stop_after_attempt(1));
        # a single 429/5xx blip would otherwise crash a multi-hour run.
        # HttpRetryOptions defaults: 5 attempts, exponential backoff with
        # jitter, retriable codes 408/429/500/502/503/504.
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(retry_options=types.HttpRetryOptions()),
        )
        self._fn_declarations: list[types.FunctionDeclaration] = []
        # None means "let the model decide" (default thinking enabled).
        # An explicit budget caps the thinking token spend per call.
        self._thinking_budget = thinking_budget

    # ------------------------------------------------------------------
    # Tool declaration
    # ------------------------------------------------------------------

    def declare_tools(self, tools: list[ToolSchema]) -> None:
        self._fn_declarations = [
            types.FunctionDeclaration(
                name=t.name,
                description=t.description,
                parameters=_clean_schema(t.parameters),
            )
            for t in tools
        ]

    # ------------------------------------------------------------------
    # Round-trip
    # ------------------------------------------------------------------

    def chat(self, history: list[Message]) -> tuple[Message, Usage]:
        system_text: str | None = None
        contents: list[types.Content] = []

        for msg in history:
            if msg.role == "system":
                # If multiple system messages exist, concatenate them.
                system_text = (system_text + "\n" + msg.text) if system_text else msg.text
            else:
                contents.append(_to_content(msg))

        config = types.GenerateContentConfig(
            system_instruction=system_text,
            tools=(
                [types.Tool(function_declarations=self._fn_declarations)]
                if self._fn_declarations else None
            ),
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                **({"thinking_budget": self._thinking_budget} if self._thinking_budget else {}),
            ),
        )

        log.debug("gemini request: %d content turns", len(contents))
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
        except genai_errors.ClientError as exc:
            # A 400 whose message indicates the per-request token maximum was
            # exceeded is a context-overflow we can stop cleanly on. Every
            # other 400 (malformed schema, bad function args, ...) is a real
            # error the caller must see -- re-raise unchanged.
            if exc.code == 400 and _is_token_limit_error(exc):
                raise ContextOverflowError(
                    getattr(exc, "message", None) or str(exc)
                ) from exc
            raise
        return _from_response(response)


# ---------------------------------------------------------------------------
# Translators
# ---------------------------------------------------------------------------


def _to_content(msg: Message) -> types.Content:
    """Translate a canonical Message to a Gemini Content object."""
    if msg.role == "tool":
        parts = [
            types.Part.from_function_response(
                name=r.name,
                # Pass payload directly; wrap only if not already a dict.
                response=r.payload if isinstance(r.payload, dict) else {"result": r.payload},
            )
            for r in msg.tool_results
        ]
        return types.Content(role="user", parts=parts)

    if msg.role == "model":
        parts: list[types.Part] = []
        # Echo back any signature that arrived on the text/thought part; Gemini
        # requires signatures wherever they appeared, not only on function-call
        # parts. The common agentic turn is tool-calls with no prose (text=None)
        # but a thought_signature -- we must still emit a carrying Part, so fall
        # back to a space (Gemini rejects empty-text parts) when a signature is
        # present without text.
        text = msg.text or (" " if msg.thought_signature else None)
        if text is not None:
            parts.append(types.Part(
                text=text,
                thought_signature=msg.thought_signature,
            ))
        for tc in msg.tool_calls:
            # Thinking models attach an opaque thought_signature to each
            # function_call Part.  The API rejects the next turn if it's absent.
            parts.append(types.Part(
                function_call=types.FunctionCall(name=tc.name, args=tc.args),
                thought_signature=tc.thought_signature,
            ))
        # Gemini rejects empty-text parts; use a space if truly empty.
        return types.Content(role=ROLE_MODEL, parts=parts or [types.Part.from_text(text=" ")])

    # user
    return types.Content(role="user", parts=[types.Part.from_text(text=msg.text or "")])


def _usage_from(response: Any) -> Usage:
    """Token accounting from usage_metadata.

    Present even on blocked / candidate-less responses — the API still
    consumed prompt tokens, so the run budget must count them.
    """
    meta = getattr(response, "usage_metadata", None)
    if not meta:
        return Usage(input_tokens=0, output_tokens=0)
    # thoughts_token_count covers chain-of-thought tokens (2.5 Pro+) which
    # are not included in candidates_token_count.
    output_tokens = int(meta.candidates_token_count or 0) + int(getattr(meta, "thoughts_token_count", None) or 0)
    return Usage(input_tokens=int(meta.prompt_token_count or 0), output_tokens=output_tokens)


def _from_response(response: Any) -> tuple[Message, Usage]:
    """Parse a Gemini GenerateContentResponse into canonical reply + usage."""
    if not response.candidates:
        # A prompt-blocked (safety-filtered) response has no candidates at
        # all. Surface it as a recoverable empty turn rather than crashing
        # the run on candidates[0].
        feedback = getattr(response, "prompt_feedback", None)
        log.error("gemini returned no candidates (prompt_feedback=%s)", feedback)
        return Message(
            role=ROLE_MODEL,
            text=f"[BLOCKED: response had no candidates; prompt_feedback={feedback}]",
        ), _usage_from(response)

    candidate = response.candidates[0]

    finish = str(candidate.finish_reason.name) if candidate.finish_reason else "UNKNOWN"
    if finish not in _OK_FINISH_REASONS:
        # SAFETY, RECITATION, MALFORMED_FUNCTION_CALL, etc. — surface as model
        # text so the orchestrator loop can react (retry, stop, log) rather than
        # silently receiving an empty message.
        log.error("gemini finish_reason=%s — returning block signal to caller", finish)
        return Message(
            role=ROLE_MODEL,
            text=f"[BLOCKED: finish_reason={finish}]",
        ), _usage_from(response)

    parts = candidate.content.parts if candidate.content else []
    text_parts: list[str] = []
    thought_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    text_thought_sig: bytes | None = None

    for part in parts:
        if part.function_call:
            fc = part.function_call
            # fc.id is set for parallel calls in some API versions; fall back
            # to a synthetic id so every ToolCall always has a unique id.
            call_id = (fc.id or None) or f"call_{uuid.uuid4().hex[:12]}"
            # fc.args is a plain dict in the google-genai SDK; json round-trip
            # ensures any nested proto Map types in future SDK versions are
            # converted to plain Python objects.
            args = json.loads(json.dumps(fc.args)) if fc.args else {}
            # thought_signature is an opaque bytes blob from the Part (not
            # FunctionCall) that must be echoed back in the next request.
            thought_sig = part.thought_signature or None
            tool_calls.append(ToolCall(id=call_id, name=fc.name, args=args, thought_signature=thought_sig))
        else:
            # Signatures can arrive on text/thought parts too, and Gemini
            # requires them echoed back wherever they appeared.
            if part.thought_signature and text_thought_sig is None:
                text_thought_sig = part.thought_signature
            if part.text and part.thought:
                thought_parts.append(part.text)
            elif part.text:
                text_parts.append(part.text)

    return Message(
        role=ROLE_MODEL,
        text="\n".join(text_parts) or None,
        reasoning="\n".join(thought_parts) or None,
        thought_signature=text_thought_sig,
        tool_calls=tool_calls,
    ), _usage_from(response)


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip JSON Schema keys Gemini doesn't support.

    Gemini supports a subset of JSON Schema including ``anyOf`` (for union
    types), but rejects ``additionalProperties``, ``$ref``, ``$schema``,
    ``allOf``, ``not``, and the conditional keywords ``if``/``then``/``else``.
    ``anyOf`` and ``oneOf`` are kept because Gemini maps them to
    ``Schema.any_of`` internally.
    """
    if not isinstance(schema, dict):
        if isinstance(schema, list):
            return [_clean_schema(v) for v in schema]
        return schema
    return {k: _clean_schema(v) for k, v in schema.items() if k not in _SCHEMA_UNSUPPORTED}
