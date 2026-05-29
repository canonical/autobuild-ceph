"""Tool dispatcher.

Maps a canonical ``ToolCall`` from the model to the corresponding handler,
catches handler errors so they surface as structured tool results (instead of
crashing the loop), and tracks the terminal ``declare_resolved`` signal.

PATTERN NOTE FOR READERS NEW TO TOOL-USING LLMS
===============================================
The dispatcher is the boundary between "what the model asked for" and "what
actually happens on the machine". The model never executes anything itself —
it just emits a structured request. This module receives that request, runs
the corresponding Python function, and returns a JSON-shaped result that
becomes part of the next conversation turn. The whole loop is just:

    model -> tool_calls -> dispatcher -> tool_results -> model -> ...

Errors must always come back as data the model can read; never as exceptions
that propagate out of here. A traceback is invisible to the model and would
just hang the loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from .. import display, guards
from ..providers.base import ToolCall, ToolResult
from .execution import ExecutionHandlers
from .filesystem import FilesystemHandlers
from .search import SearchHandlers

log = logging.getLogger(__name__)


@dataclass
class DispatchOutcome:
    results: list[ToolResult]
    declared_resolved: bool = False
    resolution_summary: str | None = None
    declared_unresolvable: bool = False
    unresolvable_reason: str | None = None


# Tools whose successful invocation counts as "the working tree changed".
# Used to clear ExecutionHandlers.files_changed_since_last_build's debt so
# the run_build hard guard knows whether a fresh build is worth doing.
_FILE_MUTATORS = {
    "write_file",
    "edit_file",
    "apply_patch",
    "delete_file",
    "replace_in_upstream",
    "drop_patch",
}


class Dispatcher:
    def __init__(
        self,
        filesystem: FilesystemHandlers,
        search: SearchHandlers,
        execution: ExecutionHandlers,
        build_log_path: str,
    ) -> None:
        self._filesystem = filesystem
        self._search = search
        self._execution = execution
        self._build_log_path = build_log_path
        # Map tool name -> bound handler. Keeping this declarative makes it
        # trivial to confirm at a glance that schema and handlers agree.
        self._handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "read_files": filesystem.read_files,
            "read_log": self._read_log_bound,
            "grep_log": self._grep_log_bound,
            "grep": search.grep,
            "git_log": search.git_log,
            "replace_in_upstream": filesystem.replace_in_upstream,
            "drop_patch": filesystem.drop_patch,
            "write_file": filesystem.write_file,
            "edit_file": filesystem.edit_file,
            "apply_patch": filesystem.apply_patch,
            "delete_file": filesystem.delete_file,
            "check_patch": execution.check_patch,
            "run_build": execution.run_build,
            "clean": execution.clean,
            # declare_resolved is handled inline so we can capture the summary
        }

    @property
    def files_changed(self) -> bool:
        """True when file mutations have occurred since the last run_build."""
        return self._execution.files_changed_since_last_build

    def _read_log_bound(self, **kwargs: Any) -> dict[str, Any]:
        # The log path lives inside the container and isn't something we want
        # the model to have to know — inject it.
        return self._filesystem.read_log(log_path=self._build_log_path, **kwargs)

    def _grep_log_bound(self, **kwargs: Any) -> dict[str, Any]:
        return self._filesystem.grep_log(log_path=self._build_log_path, **kwargs)

    def dispatch(self, calls: list[ToolCall]) -> DispatchOutcome:
        results: list[ToolResult] = []
        declared = False
        summary: str | None = None
        unresolvable = False
        unresolvable_reason: str | None = None

        for call in calls:
            log.info("dispatch %s args=%s", call.name, call.args)
            display.tool_dispatch(call.name, call.args)

            # Reject malformed args before any handler, including terminal calls.
            # A malformed declare_resolved/declare_unresolvable should be a
            # correctable error, not an irreversible loop termination.
            if "__malformed_arguments__" in call.args:
                results.append(
                    ToolResult(
                        call_id=call.id,
                        name=call.name,
                        payload={
                            "error": "your tool arguments were not valid JSON; "
                            "re-issue with valid JSON",
                            "raw": call.args["__malformed_arguments__"],
                        },
                    )
                )
                continue

            if call.name == "declare_unresolvable":
                reason = str(call.args.get("reason", ""))
                unresolvable = True
                unresolvable_reason = reason
                results.append(
                    ToolResult(
                        call_id=call.id,
                        name=call.name,
                        payload={"acknowledged": True},
                    )
                )
                continue

            if call.name == "declare_resolved":
                last = self._execution.last_build
                if last is not None and not last.ok:
                    results.append(
                        ToolResult(
                            call_id=call.id,
                            name=call.name,
                            payload={
                                "error": (
                                    "declare_resolved rejected: the last run_build "
                                    f"failed (stage={last.stage!r}, "
                                    f"rc={last.returncode}). "
                                    "The build must succeed before you can declare "
                                    "it resolved."
                                )
                            },
                        )
                    )
                    continue
                declared = True
                summary = str(call.args.get("summary", ""))
                results.append(
                    ToolResult(
                        call_id=call.id,
                        name=call.name,
                        payload={"acknowledged": True},
                    )
                )
                continue

            handler = self._handlers.get(call.name)
            if handler is None:
                results.append(
                    ToolResult(
                        call_id=call.id,
                        name=call.name,
                        payload={"error": f"unknown tool: {call.name}"},
                    )
                )
                continue

            try:
                payload = handler(**call.args)
            except guards.EditScopeViolation as exc:
                # Scope errors are *expected* — convert to a structured result
                # so the model can self-correct on the next turn.
                payload = {"error": "edit_scope_violation", "message": str(exc)}
            except TypeError as exc:
                # Wrong argument shape from the model — return as data, don't
                # crash the loop.
                payload = {"error": "bad_arguments", "message": str(exc)}
            except Exception as exc:  # noqa: BLE001
                log.exception("handler %s failed", call.name)
                payload = {"error": "handler_failure", "message": str(exc)}

            # If this call mutated the working tree, mark it so the next
            # run_build won't be refused by the hard guard.
            if call.name in _FILE_MUTATORS and payload.get("ok"):
                self._execution.files_changed_since_last_build = True

            results.append(
                ToolResult(call_id=call.id, name=call.name, payload=payload)
            )

        return DispatchOutcome(
            results=results,
            declared_resolved=declared,
            resolution_summary=summary,
            declared_unresolvable=unresolvable,
            unresolvable_reason=unresolvable_reason,
        )
