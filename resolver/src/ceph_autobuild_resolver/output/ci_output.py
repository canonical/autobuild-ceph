"""CI-oriented output for the resolver.

On success: prints the full diff (to stdout or CI_DIFF_FILE) followed by a
per-change narrative so a reviewer knows what was fixed and why.

On failure: prints a structured failure analysis — what was attempted, the
last build error, and a concrete recommendation.

Environment variables
---------------------
CI_DIFF_FILE
    If set, write the diff to this path and print only the narrative to
    stdout.  Useful when the diff is large and you want to attach it as a
    CI artefact rather than flooding the log.

CI_ANNOTATE_DIFF
    0 or 1 (default 1).  When 1, each patch section in the diff is prefixed
    with a comment block taken from the model's per-patch summary so the
    reader gets context without switching windows.

CI_MACHINE_READABLE
    0 or 1 (default 0).  When 1, emit a single JSON object to stdout and
    skip all human-readable formatting.  All other CI_* vars are ignored.
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


@dataclass
class CISuccessPayload:
    matrix_name: str
    summary: str        # model-generated free-text, may contain numbered list
    diff: str           # full unified diff from _capture_diff
    transcript_path: str


@dataclass
class CIFailurePayload:
    matrix_name: str
    stop_reason: str
    failing_command: str
    error_tail: str
    transcript_path: str


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def emit_success(payload: CISuccessPayload) -> None:
    if _machine_readable():
        _emit_json({
            "status": "success",
            "matrix_name": payload.matrix_name,
            "summary": payload.summary,
            "diff": payload.diff,
            "transcript_path": payload.transcript_path,
        })
        return

    diff_file = os.environ.get("CI_DIFF_FILE", "").strip()
    annotate = os.environ.get("CI_ANNOTATE_DIFF", "1") == "1"

    _banner(f"RESOLVED: {payload.matrix_name}")
    print()
    print("## Summary of changes")
    print()
    print(payload.summary.strip())
    print()

    if diff_file:
        with open(diff_file, "w") as fh:
            fh.write(payload.diff)
        lines = len(payload.diff.splitlines())
        print(f"## Diff written to {diff_file}  ({lines} lines)")
    else:
        print("## Diff")
        print()
        diff_body = (
            _annotate_diff(payload.diff, payload.summary) if annotate
            else payload.diff
        )
        sys.stdout.write(diff_body)
        if not diff_body.endswith("\n"):
            print()


def emit_failure(payload: CIFailurePayload) -> None:
    if _machine_readable():
        _emit_json({
            "status": "failure",
            "matrix_name": payload.matrix_name,
            "stop_reason": payload.stop_reason,
            "failing_command": payload.failing_command,
            "error_tail": payload.error_tail,
            "transcript_path": payload.transcript_path,
        })
        return

    _banner(f"UNRESOLVED: {payload.matrix_name}")
    print()
    print("## What was attempted")
    print()
    print(f"The resolver loop stopped with reason: {payload.stop_reason!r}")
    print(f"Last failing stage: {payload.failing_command!r}")
    print()
    print("## Last build error")
    print()
    print("```")
    print(payload.error_tail.strip())
    print("```")
    print()
    print("## Recommendation")
    print()
    print(_recommendation(payload.stop_reason, payload.error_tail))
    print()
    print(f"Transcript: {payload.transcript_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _banner(title: str) -> None:
    bar = "=" * 72
    print(bar)
    print(f"  {title}")
    print(bar)


def _machine_readable() -> bool:
    return os.environ.get("CI_MACHINE_READABLE", "0") == "1"


def _emit_json(obj: dict) -> None:
    json.dump(obj, sys.stdout, indent=2)
    print()


def _annotate_diff(diff: str, summary: str) -> str:
    """Prepend a summary comment above each patch section in the diff.

    Splits the diff at ``diff --git`` boundaries, looks up the patch filename
    in the model's summary text, and inserts a ``# ---`` block so each section
    is self-describing.
    """
    # Build a name→blurb lookup from the model summary.
    # Handles both numbered lists ("1. **name**: blurb") and plain mentions.
    blurbs: dict[str, str] = {}
    for m in re.finditer(
        r"\*\*([^*]+)\*\*[:\s]+([^\n]+)",
        summary,
    ):
        name = m.group(1).strip().rstrip(":")
        blurb = m.group(2).strip()
        blurbs[name] = blurb

    sections: list[str] = []
    current: list[str] = []

    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git ") and current:
            sections.append("".join(current))
            current = []
        current.append(line)
    if current:
        sections.append("".join(current))

    out: list[str] = []
    for section in sections:
        # Extract the filename from the diff header.
        m = re.search(r"^diff --git a/(\S+)", section, re.MULTILINE)
        if m:
            path = m.group(1)
            name = path.split("/")[-1]
            blurb = blurbs.get(name) or blurbs.get(path)
            if blurb:
                out.append(f"# --- {name}: {blurb}\n")
        out.append(section)

    return "".join(out)


# Map stop_reason codes to human-readable recommendations.
_RECOMMENDATIONS: dict[str, str] = {
    "validation_failed": textwrap.dedent("""\
        The fix compiled successfully but failed clean-rebuild validation —
        git apply could not re-apply the generated diff to a fresh container.

        Likely causes:
          • The diff includes files outside debian/patches/ or debian/rules
            (build artefacts, quilt .orig files).  Check _capture_diff scoping.
          • A patch hunk has wrong line-number context.  Run
            'patch -F 0 -p1 --dry-run < debian/patches/<name>.patch' on a
            clean checkout to identify which patch fails.
          • A patch deletes a file whose content in the baseline snapshot
            differs from what the diff expects (stale snapshot)."""),

    "max_iterations_exceeded": textwrap.dedent("""\
        The model used all available iterations without resolving the build.

        Likely causes:
          • The build error requires domain knowledge beyond the model's
            current system prompt.  Add a targeted invariant or example
            fix to prompts.py for this error class.
          • MAX_ITERATIONS is too low for this failure type.  Try
            MAX_ITERATIONS=50 or higher.
          • The model is looping (reading the same files without acting).
            Increase MAX_UNCHANGED_ITERATIONS or tighten the nudge messages
            in loop.py."""),

    "max_wall_seconds_exceeded": textwrap.dedent("""\
        The resolver hit the wall-clock time limit (MAX_WALL_SECONDS).
        The build itself may still be running or the model stalled.

        Try: raise MAX_WALL_SECONDS, or set MAX_SECONDS_TO_FIRST_BUILD to
        catch a model that never reaches run_build."""),

    "max_unchanged_exceeded": textwrap.dedent("""\
        The model made no file changes for several consecutive iterations.
        It is stuck in a diagnosis loop without acting.

        Try: lower MAX_UNCHANGED_ITERATIONS to force earlier nudging, or
        improve the anti-spinning rules in prompts.py (RULES YOU MUST FOLLOW
        section)."""),

    "token_budget_exceeded": textwrap.dedent("""\
        The per-run token budget was exhausted (RUN_TOKEN_BUDGET).
        Raise RUN_TOKEN_BUDGET or switch to a model with lower token usage."""),
}

_DEFAULT_RECOMMENDATION = textwrap.dedent("""\
    Inspect the transcript and the last build error above.
    Common next steps:
      • Run the resolver again with a higher iteration budget.
      • Add guidance for this error class to prompts.py.
      • If the error is a known packaging invariant, add it to the
        DEBIAN PACKAGING INVARIANTS section of the system prompt.""")


def _recommendation(stop_reason: str, error_tail: str) -> str:
    base = _RECOMMENDATIONS.get(stop_reason, _DEFAULT_RECOMMENDATION)

    # Append a targeted hint if the error tail contains a known signal.
    extra = _error_hint(error_tail)
    if extra:
        return f"{base}\n\nAdditional hint based on last error:\n  {extra}"
    return base


def _error_hint(error_tail: str) -> str:
    tail = error_tail.lower()
    if "no space left on device" in tail:
        return (
            "Disk full during build.  Free space on the LXD host "
            "(target: ≥ 40 GB free) and retry."
        )
    if "could not find boost" in tail or "boost_" in tail and "not found" in tail:
        return (
            "A Boost component is missing from BOOST_COMPONENTS or not "
            "installed.  Check if the library is header-only (no .cmake "
            "config) and remove it from BOOST_COMPONENTS if so."
        )
    if "undefined reference" in tail and "boost" in tail:
        return (
            "Linker cannot find a Boost symbol.  The component probably "
            "needs to be added to BOOST_COMPONENTS and linked explicitly."
        )
    if "does not match index" in tail:
        return (
            "git apply --index failed.  The validation container's working "
            "tree has pre-existing changes that conflict with the diff.  "
            "Ensure apply_diff resets to HEAD before applying."
        )
    if "patch" in tail and ("failed" in tail or "hunk" in tail):
        return (
            "A quilt patch failed to apply.  The line-number context in "
            "the patch is stale.  Re-derive it with replace_in_upstream."
        )
    return ""
