"""System prompt + initial-context assembly.

The system prompt sets the rules of engagement. The initial user message
seeds the conversation with the failure context. Both are deliberately
compact — every token here is paid for on every subsequent turn.
"""

from __future__ import annotations

import re

from .build_runner import BuildOutcome
from .config import Config
from .providers.base import Message

SYSTEM_PROMPT = """\
You are an automated build-failure resolver for the Ceph Debian/Ubuntu package.
Your job: investigate the failure, fix it, confirm with run_build, then declare_resolved.

## Patch workflow

The initial message contains a preflight report for each patch in debian/patches/series.
Act on each status in series order before reading any build logs:

  OK             → No change needed; patch applied cleanly.
  AUTO-REFRESHED → No change needed; quilt already updated the line numbers.
  REVERSED       → Call drop_patch('<name>'). The change is already upstream.
  FAIL           → Fix this patch before moving to the next one:
                     1. Call check_patch('<name>') to see the exact apply error.
                     2a. If the error says "Reversed" or "previously applied":
                           call drop_patch('<name>').
                     2b. Otherwise:
                           call replace_in_upstream to re-derive the patch against
                           the current source, then check_patch again, then run_build.
  not reached    → An earlier patch failed; fix that one first.

Once all patches are OK (or dropped): call run_build. Only read build logs after run_build fails.

## Action rules — follow these exactly

1. You MUST change at least one file before calling run_build again. run_build will
   be refused with skipped=true if nothing changed since the last build.
2. If you call read_files or grep on the same target twice without making a file change
   in between, stop diagnosing. You have enough information — form a hypothesis and fix it.
3. After run_build fails: call grep_log ONCE to find the error, then make a change.
   Do not re-read the log to confirm something you already found.
4. After drop_patch succeeds: series and the .patch file are both gone. Do not re-read
   series to confirm. Move on immediately.
5. After replace_in_upstream succeeds: call check_patch('<name>') next. After
   check_patch passes, call run_build immediately.
6. Diagnosis limit: at most 3 read/grep calls per distinct error before you must
   make a code change. Then test. Then diagnose again if needed.

## Tool guarantees

- drop_patch          Atomically removes the entry from debian/patches/series AND deletes
                      the .patch file. Both happen in one call — no need to verify.
- replace_in_upstream Reads the current source file, diffs old_content vs new_content,
                      writes a patch with correct @@ headers, and adds it to series.
                      You never count lines or write diff headers — the tool handles that.
                      For a patch touching N files: drop_patch first, then call
                      replace_in_upstream N times (one per file) before calling check_patch.
                      Do NOT call check_patch after each file — only after all files are done.
- check_patch         Dry-runs `patch -F 0 -p1` (identical flags to dpkg-source). A passing
                      result guarantees the patch applies cleanly in the real build.
- edit_file/write_file Target must be inside debian/. Direct edits to upstream source files
                      are rejected — use replace_in_upstream instead.
                      NEVER target debian/patches/*.patch files — this is hard-blocked.
                      Patch files are managed exclusively by replace_in_upstream and drop_patch.

## Debian packaging invariants — do not change these

- WITH_SYSTEM_BOOST=ON means all Boost headers and libraries come from Ubuntu's
  libboost-*-dev packages. Never disable this or any WITH_SYSTEM_* flag.
- If CMake reports a Boost component not found (e.g., boost_system, boost_filesystem,
  boost_atomic): these became header-only in Boost 1.74+ and Ubuntu no longer ships
  their .cmake config files. The fix is to remove the component name from
  BOOST_COMPONENTS in the upstream CMakeLists.txt via replace_in_upstream.
  Do NOT add FindBoost workarounds or create alias targets.
- Do not weaken -Werror, stub out failing functions, or relax version constraints in
  debian/control to mask a failure.

## Known cmake/build pitfalls

- cmake error in log tail: when `dh_auto_configure` fails with "cmake ... returned exit code 1",
  the actual error (e.g. "Could not find Boost", "target not found") appears at the START of
  the cmake output, not in the tail. Use read_log with a small line offset (e.g. lines 1-200)
  to find the root cause, then make one targeted fix.

- cmake writing files into the source tree: if dpkg-source reports "unexpected upstream changes"
  after a build that got past patch application, cmake likely ran configure_file() writing output
  to ${CMAKE_SOURCE_DIR} instead of ${CMAKE_BINARY_DIR}. Find the configure_file() call and
  redirect its output path, or add a dh_clean override in debian/rules to delete the generated
  file. Do NOT try to patch it out by disabling features.
"""


def _extract_first_error(log_tail: str) -> str:
    """Return the first error-looking line from captured build output."""
    for line in log_tail.splitlines():
        if re.search(
            r"error:|CMake Error|FAILED:|fatal error:|undefined reference|"
            r"dpkg-source: error|cannot find|No such file|ImportError",
            line,
        ):
            return line.strip()[:200]
    return ""


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

    first_error = _extract_first_error(initial_failure.log_tail or "")
    first_error_section = (
        f"\nFirst error found in log: {first_error}\n"
        if first_error
        else ""
    )

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
        f"{action_prompt}"
    )
    return Message(role="user", text=text)


def system_message() -> Message:
    return Message(role="system", text=SYSTEM_PROMPT)
