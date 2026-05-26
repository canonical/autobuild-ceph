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
from google import genai
from google.genai import types

from .base import Message, ProviderAdapter, ToolCall, ToolSchema, Usage

log = logging.getLogger(__name__)

# finish_reason values that indicate the model stopped normally.
_OK_FINISH_REASONS = frozenset({"STOP", "MAX_TOKENS", "FINISH_REASON_UNSPECIFIED"})
_SCHEMA_UNSUPPORTED = frozenset({
    "$schema", "$id", "$ref",
    "additionalProperties", "unevaluatedProperties",
    "allOf", "not", "if", "then", "else",
})


class GeminiAdapter(ProviderAdapter):
    def __init__(self, api_key: str, model: str) -> None:
        self._model = model
        self._client = genai.Client(api_key=api_key)
        self._fn_declarations: list[types.FunctionDeclaration] = []

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
        )

        log.debug("gemini request: %d content turns", len(contents))
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
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
        if msg.text:
            parts.append(types.Part.from_text(text=msg.text))
        for tc in msg.tool_calls:
            # Thinking models attach an opaque thought_signature to each
            # function_call Part.  The API rejects the next turn if it's absent.
            parts.append(types.Part(
                function_call=types.FunctionCall(name=tc.name, args=tc.args),
                thought_signature=tc.thought_signature,
            ))
        # Gemini rejects empty-text parts; use a space if truly empty.
        return types.Content(role="model", parts=parts or [types.Part.from_text(text=" ")])

    # user
    return types.Content(role="user", parts=[types.Part.from_text(text=msg.text or "")])


def _from_response(response: Any) -> tuple[Message, Usage]:
    """Parse a Gemini GenerateContentResponse into canonical reply + usage."""
    candidate = response.candidates[0]

    finish = str(candidate.finish_reason.name) if candidate.finish_reason else "UNKNOWN"
    if finish not in _OK_FINISH_REASONS:
        # SAFETY, RECITATION, MALFORMED_FUNCTION_CALL, etc. — surface as model
        # text so the orchestrator loop can react (retry, stop, log) rather than
        # silently receiving an empty message.
        log.error("gemini finish_reason=%s — returning block signal to caller", finish)
        return Message(
            role="model",
            text=f"[BLOCKED: finish_reason={finish}]",
        ), Usage(input_tokens=0, output_tokens=0)

    parts = candidate.content.parts if candidate.content else []
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

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
        elif part.text and not part.thought:
            # Skip thought=True parts (internal reasoning); only collect output text.
            text_parts.append(part.text)

    meta = response.usage_metadata
    if meta:
        # thoughts_token_count covers chain-of-thought tokens (2.5 Pro+) which
        # are not included in candidates_token_count.
        output_tokens = int(meta.candidates_token_count or 0) + int(getattr(meta, "thoughts_token_count", None) or 0)
        input_tokens = int(meta.prompt_token_count or 0)
    else:
        input_tokens = output_tokens = 0

    return Message(
        role="model",
        text="\n".join(text_parts) or None,
        tool_calls=tool_calls,
    ), Usage(input_tokens=input_tokens, output_tokens=output_tokens)


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
