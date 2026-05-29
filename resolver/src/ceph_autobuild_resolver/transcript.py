"""Append-only JSONL log of every interaction with the model.

This is the artefact the PR attaches for human review. Every model turn,
every tool call, every tool result lands here as a JSON line so a reviewer
can step through the resolution session.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .providers.base import Message, Usage


class Transcript:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate at start: each resolver run owns its transcript.
        self._path.write_text("")

    def _write(self, kind: str, **fields: Any) -> None:
        record = {"ts": time.time(), "kind": kind, **fields}
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def initial_context(self, summary: dict[str, Any]) -> None:
        self._write("initial_context", summary=summary)

    def preflight(
        self,
        *,
        dropped: list[str],
        refreshed: list[str],
        report: str,
    ) -> None:
        """Record the deterministic preflight pass for PR-review traceability."""
        self._write(
            "preflight",
            dropped=dropped,
            refreshed=refreshed,
            report=report,
        )

    def model_turn(self, msg: Message, usage: Usage) -> None:
        self._write(
            "model_turn",
            text=msg.text,
            reasoning=msg.reasoning,
            tool_calls=[asdict(tc) for tc in msg.tool_calls],
            usage=asdict(usage),
        )

    def tool_results(self, results: list[Any]) -> None:
        self._write("tool_results", results=[asdict(r) for r in results])

    def build_attempt(self, outcome: Any) -> None:
        self._write(
            "build_attempt",
            ok=outcome.ok,
            stage=outcome.stage,
            returncode=outcome.returncode,
            log_path=outcome.log_path,
        )

    def validation(self, ok: bool, stage: str, returncode: int) -> None:
        self._write("validation", ok=ok, stage=stage, returncode=returncode)

    def outcome(self, kind: str, **fields: Any) -> None:
        self._write("outcome", outcome_kind=kind, **fields)

    @property
    def path(self) -> Path:
        return self._path
