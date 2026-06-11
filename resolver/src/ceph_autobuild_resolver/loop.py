"""The resolution loop.

One turn through the loop:

1. Send the conversation history to the provider.
2. Receive a model reply (possibly with tool calls).
3. Append the reply to history.
4. Dispatch any tool calls; collect structured results.
5. Append a synthetic ``tool``-role message carrying those results.
6. Update the budget; check capacity.
7. If the model called ``declare_resolved``, return so the orchestrator can
   run the validation pass. Otherwise repeat.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from . import display
from .budget import Budget
from .build_runner import first_error_line
from .providers.base import (
    ROLE_MODEL,
    ContextOverflowError,
    Message,
    ProviderAdapter,
)
from .tools.dispatch import Dispatcher
from .transcript import Transcript

log = logging.getLogger(__name__)


def _maybe_record_compilation(budget: Budget, payload: dict) -> None:
    """Mark compilation reached if this run_build result cleared dpkg-source.

    Signals used (in priority order):
      1. ok=True  — full build succeeded, obviously cleared patching.
      2. log_tail has no "dpkg-source: error" — we got past patch application
         and into cmake/ninja/make, even if compilation itself failed.
    """
    if payload.get("ok"):
        budget.record_compilation_reached()
        return
    tail = payload.get("log_tail") or ""
    if tail and "dpkg-source: error" not in tail:
        budget.record_compilation_reached()


# Compress history when it exceeds this many messages, retaining only the
# most recent _KEEP_RECENT_TURNS entries verbatim plus a summary of the rest.
# Bounds per-call input tokens to O(n) rather than O(n²) over the loop.
#
# Observed growth is ~500-800 tok/turn (reasoning is discarded from history,
# only tool calls and text accumulate). The original values (24/10) compressed
# every ~5 model turns, far too often for multi-file patch tasks. With 80/30:
# context oscillates 17K-57K, compression fires every ~23 turns, and the model
# retains 15 full turn-pairs of visible context after each compression.
# Safe for MAX_ITERATIONS=1000: ~43 compressions total, never close to limits.
_COMPRESS_AFTER_TURNS = 80
_KEEP_RECENT_TURNS = 30


def _summarize_turns(turns: list[Message]) -> str:
    """Build a compact text summary of middle turns for history compression."""
    entries: list[str] = []

    for msg in turns:
        if msg.role == ROLE_MODEL:
            for tc in msg.tool_calls:
                if tc.name == "run_build":
                    entries.append("• run_build called")
                elif tc.name in ("write_file", "edit_file"):
                    entries.append(f"• edited {tc.args.get('path', '?')}")
                elif tc.name == "apply_patch":
                    entries.append(f"• applied patch {tc.args.get('patch_path', '?')}")
                elif tc.name == "replace_in_upstream":
                    entries.append(f"• upstream change: {tc.args.get('source_path', '?')}")
                elif tc.name == "drop_patch":
                    entries.append(f"• dropped patch {tc.args.get('patch_name', '?')}")
                elif tc.name == "delete_file":
                    entries.append(f"• deleted {tc.args.get('path', '?')}")
        elif msg.role == "tool":
            for r in msg.tool_results:
                if r.name == "run_build":
                    ok = r.payload.get("ok", False)
                    stage = r.payload.get("stage", "?")
                    rc = r.payload.get("returncode", "?")
                    status = "SUCCEEDED" if ok else f"FAILED (stage={stage}, rc={rc})"
                    entries.append(f"  → {status}")
                    if not ok:
                        tail = r.payload.get("log_tail", "") or ""
                        for line in tail.splitlines():
                            if any(
                                k in line.lower()
                                for k in ("error:", "fatal:", "undefined reference")
                            ):
                                entries.append(f"    {line.strip()[:150]}")
                                break
                elif r.name == "check_patch":
                    ok = r.payload.get("ok", False)
                    patch = r.payload.get("patch", "?")
                    status = "PASSED" if ok else "FAILED"
                    entries.append(f"  → check_patch {patch}: {status}")

    if not entries:
        return "[Session history compressed — earlier turns had no file changes or builds]"
    return (
        "[Session history compressed — earlier actions summarized]\n\n"
        + "\n".join(entries)
    )


def _compress_history(history: list[Message]) -> None:
    """Replace middle turns with a compact summary, mutating history in place.

    Always preserves history[0] (system prompt) and history[1] (initial user
    context) plus the _KEEP_RECENT_TURNS most recent messages. Everything in
    between is collapsed into a single synthetic summary message.

    INVARIANT: history[0] must be the system message and history[1] must be
    the initial user context message (set by orchestrator before run_loop is
    called). The ``cut = max(cut, 2)`` and ``del history[2:cut]`` below depend
    on this layout.
    """
    if len(history) <= _COMPRESS_AFTER_TURNS:
        return

    # Walk cut backward until we land on a non-tool message so we never split
    # a model-turn / tool-results pair.
    cut = len(history) - _KEEP_RECENT_TURNS
    cut = max(cut, 2)
    while cut > 2 and history[cut].role == "tool":
        cut -= 1

    if cut <= 2:
        return

    middle = history[2:cut]
    summary = Message(role="user", text=_summarize_turns(middle))
    del history[2:cut]
    history.insert(2, summary)
    log.info(
        "compressed history: replaced %d middle turns with summary (%d total)",
        len(middle),
        len(history),
    )


@dataclass
class LoopResult:
    declared_resolved: bool
    resolution_summary: str | None
    history: list[Message] = field(default_factory=list)
    stop_reason: str = "ok"


def run_loop(
    *,
    history: list[Message],
    provider: ProviderAdapter,
    dispatcher: Dispatcher,
    budget: Budget,
    transcript: Transcript,
) -> LoopResult:
    """Drive the conversation until the model resolves or the budget runs out.

    ``history`` is mutated in place — caller can inspect it afterwards (e.g.
    to re-enter the loop after a validation failure).
    """
    # First error line from the most recent failed run_build, for nudge messages.
    last_build_error: str = ""

    while budget.has_capacity():
        _compress_history(history)
        budget.record_iteration()

        try:
            reply, usage = provider.chat(history)
        except ContextOverflowError as exc:
            # A single request exceeded the provider's input-token limit. We
            # can't recover by compressing: the offending message is the most
            # recent one (inside the keep-recent window), so compression can't
            # evict it and a retry would just overflow again. Stop cleanly so
            # the orchestrator records the outcome instead of crashing.
            log.error("context overflow from provider, stopping: %s", exc)
            return LoopResult(
                declared_resolved=False,
                resolution_summary=None,
                history=history,
                stop_reason="context_overflow",
            )
        except Exception as exc:  # noqa: BLE001
            # Any other provider failure has already exhausted the adapter's
            # own retries (e.g. persistent 429/5xx). Stop cleanly so the
            # orchestrator records an outcome and emits a CI artifact rather
            # than losing the whole run to a raw traceback.
            log.exception("provider error, stopping: %s", exc)
            return LoopResult(
                declared_resolved=False,
                resolution_summary=f"provider error: {exc}",
                history=history,
                stop_reason="provider_error",
            )
        budget.record_usage(usage)
        display.token_usage(budget.input_tokens_used, budget.output_tokens_used, budget.cfg.run_token_budget)
        transcript.model_turn(reply, usage)
        history.append(reply)
        if reply.reasoning:
            display.model_reasoning(reply.reasoning)
        if reply.text:
            display.model_text(reply.text)

        if not reply.tool_calls:
            # No tool calls means the model replied with prose only. That's
            # not a productive turn for us; nudge it to use tools or stop.
            log.warning(
                "model returned no tool calls (text=%r); nudging",
                (reply.text or "")[:200],
            )
            history.append(
                Message(
                    role="user",
                    text=(
                        "Use the available tools to investigate or apply "
                        "fixes. If you believe the build is fixed, run_build "
                        "and then declare_resolved."
                    ),
                )
            )
            continue

        try:
            outcome = dispatcher.dispatch(reply.tool_calls)
        except BaseException as exc:
            # Catch anything that escaped the dispatcher's own broad handler
            # (e.g. BrokenPipeError from display, or a transient LXD fault).
            # Surface it as a structured tool error so the loop can continue
            # rather than silently dying without writing to the transcript.
            log.exception("dispatcher raised unexpectedly: %s", exc)
            from .providers.base import ToolResult
            from .tools.dispatch import DispatchOutcome, clamp_oversized_results
            synthetic_results = [
                ToolResult(
                    call_id=tc.id,
                    name=tc.name,
                    payload={"error": "dispatcher_crash", "message": str(exc)},
                )
                for tc in reply.tool_calls
            ]
            # str(exc) is normally small, but this is the one result-append
            # path outside dispatch(); run it through the same byte ceiling so
            # a pathological exception message can't overflow the next request.
            clamp_oversized_results(synthetic_results)
            outcome = DispatchOutcome(results=synthetic_results)
        transcript.tool_results(outcome.results)
        history.append(Message(role="tool", tool_results=outcome.results))

        if outcome.declared_unresolvable:
            log.warning(
                "model declared unresolvable: %s", outcome.unresolvable_reason
            )
            return LoopResult(
                declared_resolved=False,
                resolution_summary=outcome.unresolvable_reason,
                history=history,
                stop_reason="declared_unresolvable",
            )

        if outcome.declared_resolved:
            return LoopResult(
                declared_resolved=True,
                resolution_summary=outcome.resolution_summary,
                history=history,
                stop_reason="resolved",
            )

        # If check_patch just passed and there are pending file changes (i.e.
        # run_build would execute if called now), nudge the model to do it.
        # dispatcher.files_changed is the authoritative "would run_build execute?"
        # signal — it reflects execution.files_changed_since_last_build which is
        # set by the dispatcher on every successful file-mutation tool call.
        if dispatcher.files_changed:
            for r in outcome.results:
                if r.name == "check_patch" and r.payload.get("ok"):
                    history.append(
                        Message(
                            role="user",
                            text=(
                                "check_patch passed — the patch applies cleanly. "
                                "Call run_build now to verify it actually compiles."
                            ),
                        )
                    )
                    break

        for r in outcome.results:
            if r.name == "run_build":
                if not r.payload.get("skipped") and not budget.compilation_reached:
                    _maybe_record_compilation(budget, r.payload)
                # Track the first error from every failed build so nudge
                # messages can be specific about what needs fixing.
                if not r.payload.get("ok") and not r.payload.get("skipped"):
                    tail = r.payload.get("log_tail") or ""
                    line = first_error_line(tail)
                    if line:
                        last_build_error = line

                if r.payload.get("skipped"):
                    # The hard guard refused the build: nothing changed since
                    # the previous failed attempt. This is the authoritative
                    # no-progress signal — the execution layer already enforced
                    # it, so we don't need a separate local flag.
                    budget.record_unchanged_build()
                    error_hint = (
                        f"\nLast known error: {last_build_error}" if last_build_error else ""
                    )
                    history.append(
                        Message(
                            role="user",
                            text=(
                                "run_build was REFUSED (no_changes_since_last_build). "
                                "You MUST change at least one file before calling run_build. "
                                "Use edit_file, write_file, replace_in_upstream, or "
                                "drop_patch to make a fix, then call run_build."
                                f"{error_hint}"
                            ),
                        )
                    )
                    continue
                # Build executed (not skipped): file changes were present (or
                # this is the first build). Either way reset the streak.
                budget.reset_unchanged_streak()
                if r.payload.get("ok"):
                    last_build_error = ""

    return LoopResult(
        declared_resolved=False,
        resolution_summary=None,
        history=history,
        stop_reason=budget.reason_for_stop(),
    )
