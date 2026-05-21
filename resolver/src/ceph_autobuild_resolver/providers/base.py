"""Canonical types and the ``ProviderAdapter`` protocol.

WHY THIS LAYER EXISTS
=====================

Every LLM provider that supports "function calling" / "tool use" exposes the
same conceptual feature, but uses a slightly different wire format:

* OpenAI / OpenRouter use a list of ``tools`` with ``type: "function"`` and
  arguments returned as **JSON-encoded strings**.
* Google's Gemini uses ``function_declarations`` and returns arguments as
  already-parsed objects.
* Roles, tool-result attachment, and usage-counter field names all differ.

Rather than smear that conditional logic throughout the resolution loop, the
loop talks only to the *canonical* dataclasses defined here. A thin per-provider
adapter (``providers/openrouter.py``, ``providers/gemini.py``) is responsible
for translating between the canonical form and the wire form in both
directions. Adding a third provider is a new adapter file — nothing else
changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

# ---------------------------------------------------------------------------
# Canonical message / tool-call types
# ---------------------------------------------------------------------------
#
# In a chat-style LLM API a "conversation" is a list of messages, each tagged
# with a role: "system" (preamble / instructions), "user" (input we send),
# "model" (assistant reply), or "tool" (the *result* of a tool call we
# previously executed on behalf of the model). Tool calls flow as part of the
# model's reply; tool *results* flow back as their own messages so the model
# can read them on its next turn.
#
# We use the role name "model" internally (Gemini's term). The OpenRouter
# adapter rewrites it to "assistant" on the wire.

Role = Literal["system", "user", "model", "tool"]


@dataclass
class ToolCall:
    """A single function/tool invocation produced by the model.

    ``id`` is provider-assigned (OpenRouter calls it ``tool_call_id``; Gemini
    doesn't strictly require one but we synthesise one so tool results can
    always be attached to the call that produced them).

    ``args`` is **already deserialized** to a Python dict — the adapter is
    responsible for parsing the JSON string OpenAI-style providers return.
    Downstream code never has to remember which provider it's running against.
    """

    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ToolResult:
    """The outcome of executing a ``ToolCall``, sent back on the next turn.

    ``call_id`` matches ``ToolCall.id`` so the model can correlate request and
    response. ``payload`` is whatever JSON-serialisable structure the tool
    handler produced (e.g. file contents, grep matches, error info).
    """

    call_id: str
    name: str
    payload: dict[str, Any]


@dataclass
class Message:
    """A canonical conversation entry.

    A single Message can carry text, tool calls, or tool results, depending on
    role:

    * system / user: ``text`` populated.
    * model: ``text`` and/or ``tool_calls`` populated. The model can produce
      multiple tool calls in one turn (a "parallel" tool call) and the loop
      dispatches them all before the next round-trip.
    * tool: ``tool_results`` populated. Multiple results may be batched into
      one message for symmetry with parallel tool calls.
    """

    role: Role
    text: str | None = None
    # Reasoning summary the provider exposed for this turn. Captured for
    # display and transcript only; never forms part of the prompt.
    reasoning: str | None = None
    # Opaque reasoning blocks (provider-specific shape) that MUST be echoed
    # back on subsequent turns when the model used thinking + tool calls.
    # Anthropic and Gemini both reject the next turn otherwise.
    reasoning_details: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass
class Usage:
    """Token accounting in a provider-neutral shape.

    Each provider reports tokens slightly differently (OpenRouter:
    ``prompt_tokens``/``completion_tokens``; Gemini: ``promptTokenCount``/
    ``candidatesTokenCount``). Adapters normalise to this struct so the
    budget tracker doesn't have to care.
    """

    input_tokens: int
    output_tokens: int


@dataclass
class ToolSchema:
    """Provider-neutral declaration of a tool the model is allowed to call.

    ``parameters`` is a JSON Schema object describing the arguments. Both
    OpenRouter and Gemini accept JSON Schema (with minor dialect differences),
    so the adapter typically passes this through with light reshaping.
    """

    name: str
    description: str
    parameters: dict[str, Any]


# ---------------------------------------------------------------------------
# Protocol every adapter implements
# ---------------------------------------------------------------------------


class ProviderAdapter(Protocol):
    """Round-trip a canonical conversation through a specific provider."""

    def declare_tools(self, tools: list[ToolSchema]) -> None:
        """Register the tool catalogue. Called once at startup."""
        ...

    def chat(self, history: list[Message]) -> tuple[Message, Usage]:
        """Send the conversation, return the model's reply and usage.

        Implementations MUST:

        * Translate ``history`` into the provider's wire format (role names,
          tool-result attachment style, system-prompt placement).
        * Parse the response into a single canonical ``Message`` with
          ``role="model"``. If the model produced tool calls, populate
          ``tool_calls`` with arguments **already deserialized** to dicts.
        * Return token counts in canonical ``Usage`` form.
        """
        ...
