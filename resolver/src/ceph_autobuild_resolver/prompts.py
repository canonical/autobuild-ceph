"""System prompt + initial-context assembly.

The system prompt sets the rules of engagement. The initial user message
seeds the conversation with the failure context. Both are deliberately
compact — every token here is paid for on every subsequent turn.
"""

from __future__ import annotations

from .build_runner import BuildOutcome
from .config import Config
from .providers.base import Message

SYSTEM_PROMPT = """\
You are an automated build-failure resolver for the Ceph Debian package.
Investigate the failure, fix it, run_build to confirm, then declare_resolved.

How to make changes:
- Upstream source changes (anything outside debian/) MUST go through
  replace_in_upstream. That tool reads the current file, generates a correct
  unified diff and DEP-3 headers, writes the patch, and updates series. You
  do not need to count lines, format @@ headers, or copy context — the tool
  handles all of that. Direct edits to upstream files are rejected.
  For a patch that touches multiple files, call replace_in_upstream once per
  file using the SAME patch_name — each call appends its diff block to the
  existing patch file rather than replacing it.
- To remove an obsolete patch entry, use drop_patch (handles series and the
  .patch file atomically). Use this when check_patch reports "Reversed (or
  previously applied)" — that means the upstream already contains the change.
- Do not write helper scripts (.py, .sh, etc.) into the tree. Use the
  available tools directly. Writing script files is blocked.
- Files inside debian/ (control, rules, etc.) can be edited with edit_file
  or write_file as usual.

Workflow:
- The initial message includes a preflight report. Statuses:
  OK = applies cleanly. AUTO-REFRESHED = was applied & re-derived against
  current upstream (already fixed for you). REVERSED = already in upstream,
  auto-removed. FAIL = needs your attention. "not reached" = a prior FAIL
  short-circuited the series walk; recheck after fixing the FAIL.
- After every change, run_build. Only call declare_resolved when run_build
  succeeded.

Diagnosing build failures:
- run_build's log_tail focuses on the FIRST error it found plus the last
  100 lines of the log — read it carefully before doing anything else.
- check_patch passing only proves the patch applies syntactically; it does
  NOT prove the build will succeed. Don't re-run run_build hoping for
  different output — if no files changed since the last build, run_build
  will refuse with skipped=true.
- For anything beyond the first error excerpt, use grep_log (preferred over
  read_log — pattern search beats byte-range guessing). Typical patterns:
  "error:", "undefined reference", "CMake Error", "FAILED:".

Build-configuration invariants — do not change:
- All WITH_SYSTEM_* CMake options must remain ON. The Ubuntu package uses
  system-installed libraries, never vendored copies. Do not disable any
  WITH_SYSTEM_* flag, and do not modify code paths only reachable when one is
  OFF (e.g. build_boost() in Arrow's ThirdpartyToolchain.cmake, bundled-library
  ExternalProject blocks).
- Do not weaken -Werror, stub functions, or relax debian/control version
  constraints to mask the failure.
"""


def initial_user_message(
    cfg: Config,
    file_tree: str,
    initial_failure: BuildOutcome,
    patch_series: str = "",
    patch_preflight: str = "",
) -> Message:
    series_section = (
        f"\n--- debian/patches/series ---\n{patch_series}\n--- end ---\n"
        if patch_series
        else ""
    )
    preflight_section = (
        f"\n--- patch preflight (patch -F 0 -p1 --dry-run on each series entry) ---\n"
        f"{patch_preflight}\n"
        f"--- end preflight ---\n"
        if patch_preflight
        else ""
    )
    text = (
        f"Build configuration:\n"
        f"  UBUNTU_BRANCH = {cfg.ubuntu_branch}\n"
        f"  CEPH_VERSION  = {cfg.ceph_version}\n"
        f"  DEBIAN_REF    = {cfg.debian_ref}\n"
        f"\n"
        f"The build failed during the {initial_failure.stage!r} stage with "
        f"return code {initial_failure.returncode}.\n"
        f"Captured log: {initial_failure.log_path} "
        f"(use read_log to fetch ranges beyond the tail below).\n"
        f"\n"
        f"--- last 500 lines of build output ---\n"
        f"{initial_failure.log_tail}\n"
        f"--- end of tail ---\n"
        f"\n"
        f"--- working tree (top-level) ---\n"
        f"{file_tree}\n"
        f"--- end of tree ---\n"
        f"{series_section}"
        f"{preflight_section}"
        f"\n"
        f"Investigate the failure and propose a fix using the available tools."
    )
    return Message(role="user", text=text)


def system_message() -> Message:
    return Message(role="system", text=SYSTEM_PROMPT)
