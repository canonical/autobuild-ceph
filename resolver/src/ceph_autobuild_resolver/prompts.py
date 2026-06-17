"""System prompt + initial-context assembly.

The system prompt sets the rules of engagement. The initial user message
seeds the conversation with the failure context. Both are deliberately
compact — every token here is paid for on every subsequent turn.
"""

from __future__ import annotations

import uuid

from .build_runner import BuildOutcome, first_error_line
from .config import Config
from .providers.base import Message

SYSTEM_PROMPT = """\
You are an automated build-failure resolver for the Ceph Debian/Ubuntu package.
Your job: investigate the failure, fix it, confirm with run_build, then declare_resolved.

## Phase 1 — patch series (do this first)

The initial message has a preflight report for each patch in debian/patches/series.
Act on each status in series order BEFORE reading any build logs. Repairing the series
needs several patch fixes before the first build — that is expected.

  OK / AUTO-REFRESHED → No change needed.
  REVERSED            → drop_patch('<name>'). The change is already upstream.
  FAIL                → check_patch('<name>') to see the apply error, then:
                          - "Reversed"/"previously applied" → drop_patch('<name>').
                          - otherwise → replace_in_upstream to re-derive against current
                            source, then check_patch again.
  not reached         → An earlier patch failed; fix that one first.

Once every patch is OK or dropped: call run_build.

## Phase 2 — the build loop (smallest change, then build)

- Make the SMALLEST change that could fix the CURRENT error, then run_build. Do NOT
  batch several fixes between builds. One change → one build → read the new error.
- Each run_build may surface a DIFFERENT error. When it does, treat it as a fresh
  problem: re-read the new first-error line and re-decide. Do NOT keep extending a
  patch you started for a previous error — that is how you get lost.
- If you have made 3+ file changes without a run_build, build now.
- Diagnose cheaply. grep to find the relevant lines, then read_files a TIGHT window
  (start_line/end_line, <=80 lines). Do NOT read whole large source files — it crowds
  out your reasoning and rarely helps. Don't read the same target twice without a
  change in between; you already have enough — act.
- After run_build fails: grep_log ONCE for the error, then make a change.

## Common fixes — check these BEFORE any large or multi-file patch

1. Boost component not found. Symptom: CMake "Could NOT find Boost (missing: <X>)"
   while Ubuntu's libboost-*-dev is installed. The component's CMake config is not
   shipped. First try: ONE replace_in_upstream removing <X> from the BOOST_COMPONENTS
   list in the CMakeLists.txt the error names — a one-line, one-file change. Build to
   confirm; if a later link error shows <X> is genuinely required, re-anchor on that.
   Do NOT add FindBoost workarounds, alias targets, or rewrite Boost usages in source.
   (Don't reason about which components are header-only — let the build decide.)
   WITH_SYSTEM_BOOST stays ON.

2. Error in an OPTIONAL subsystem (a language binding, dashboard plugin, gateway
   extension, experimental component). Fix: disable it with -DWITH_<FEATURE>=OFF in
   debian/rules before attempting source patches (see decision tree Step 1). Core
   daemons — radosgw, cephfs, rbd, mon, osd — are NOT optional.

3. dpkg-source "unexpected upstream changes" after patches applied: cmake ran
   configure_file() writing into ${CMAKE_SOURCE_DIR}. Redirect it to
   ${CMAKE_BINARY_DIR}, or add a dh_clean override in debian/rules to delete the
   generated file. Do NOT disable features for this.

## When to change strategy (decision tree)

Use this when 3+ attempts on the same error fail, OR the build output keeps changing
without getting closer (making the error "different" is not progress), OR you're about
to write a large/multi-file patch.

Step 0 — Is it a Common fix above? Apply it. Most configure errors are #1 or #2.
Step 1 — Is the failing code optional/feature-gated (inside #ifdef WITH_*, or a
  WITH_*-gated cmake target) and not a core daemon? grep CMakeLists.txt for the
  WITH_<FEATURE> option, add -DWITH_<FEATURE>=OFF to the cmake call in debian/rules,
  and drop any now-unbuilt packages from debian/control plus their install steps.
Step 2 — Wrong layer? Prefer an upstream compatibility header or a cmake find-module
  over hand-rolled namespace aliases, include-path edits, or typedefs. Fix the root
  cause, not the symptom.
Step 3 — Only after (a) confirming it is none of the Common fixes and (b) a run_build
  that tested your most recent change: declare_unresolvable, stating what you tried,
  why each failed, and what information is missing. Never declare without a build
  behind it.

## Hard constraints

- run_build is refused (skipped=true) unless at least one file changed since the last
  build. drop_patch is atomic (removes the series entry AND deletes the .patch file) —
  no need to verify. check_patch dry-runs a patch against the tree in its CURRENT
  state; it does not guarantee dpkg-source's series-order application succeeds — make
  sure earlier patches in series are applied (or accounted for) before trusting it.
- edit_file/write_file target debian/ ONLY; never .patch files (managed exclusively by
  replace_in_upstream / drop_patch). Upstream source is changed only via
  replace_in_upstream.
- A single patch may legitimately span multiple files: call replace_in_upstream once
  per file with the SAME patch_name, then check_patch ONCE, then run_build. This is the
  exception — first confirm a one-file Common fix won't do.
- Do not weaken -Werror, stub failing functions, or relax debian/control version
  constraints to mask a failure. Never disable any WITH_SYSTEM_* flag.

## Reading the build log

The run_build excerpt anchors on the real "CMake Error" / first compile "error:" and
the file:line it names — read THAT, not the surrounding "-- Looking for ... /
-- Performing Test ... - Failed" probe output, which is benign. Use read_log directly
only if the excerpt is truncated.
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

    first_error = first_error_line(initial_failure.log_tail or "")
    first_error_section = (
        f"\nFirst error found in log: {first_error}\n"
        if first_error
        else ""
    )

    # Fence the (build-controlled) log tail with an unguessable per-call
    # boundary. A fixed delimiter like "--- end of tail ---" can appear verbatim
    # in a build log (a diff, a script echoing separators) and let crafted log
    # content close the fence early and inject text that looks like first-party
    # instructions. The model can't see this token ahead of time, so the log
    # body can't reproduce it.
    boundary = uuid.uuid4().hex

    # Direct the model to the right starting point based on what we know.
    has_fail_patch = "FAIL" in (patch_preflight or "")
    if has_fail_patch:
        action_prompt = (
            "The preflight shows one or more FAIL patches. "
            "Your first action: fix each FAIL patch in series order using the patch "
            "workflow from the system prompt. Do NOT read build logs yet — "
            "sort out the patch series first, then call run_build."
        )
    else:
        action_prompt = (
            "Investigate the build failure and propose a fix using the available tools."
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
        f"{first_error_section}"
        f"\n"
        f"--- begin build output [{boundary}] ---\n"
        f"{initial_failure.log_tail}\n"
        f"--- end build output [{boundary}] ---\n"
        f"\n"
        f"--- working tree (top-level) ---\n"
        f"{file_tree}\n"
        f"--- end of tree ---\n"
        f"{series_section}"
        f"{preflight_section}"
        f"\n"
        f"{action_prompt}"
    )
    return Message(role="user", text=text)


def system_message() -> Message:
    return Message(role="system", text=SYSTEM_PROMPT)
