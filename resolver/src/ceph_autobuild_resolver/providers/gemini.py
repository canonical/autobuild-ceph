"""Gemini adapter — placeholder.

We're starting integration testing with OpenRouter. This module exists so the
provider-selection plumbing has somewhere to land when we add Gemini support;
the canonical types in ``base`` already model both providers.

Implementation will use ``google-genai`` (or REST against
``generativelanguage.googleapis.com``). Notable translation differences from
OpenRouter:

* Tools declared as ``function_declarations`` (no ``type: function`` wrapper).
* Roles are ``user`` / ``model`` (no ``assistant``).
* Tool results attach as ``functionResponse`` parts inside a ``user`` message,
  not as a standalone role.
* Tool-call arguments come back already parsed — no ``json.loads`` needed.
* Token counts: ``promptTokenCount`` / ``candidatesTokenCount``.
"""

from __future__ import annotations

from .base import Message, ProviderAdapter, ToolSchema, Usage


class GeminiAdapter(ProviderAdapter):
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def declare_tools(self, tools: list[ToolSchema]) -> None:
        raise NotImplementedError("Gemini adapter not yet implemented")

    def chat(self, history: list[Message]) -> tuple[Message, Usage]:
        raise NotImplementedError("Gemini adapter not yet implemented")
