"""Build stage execution inside an LXD container.

Each public method builds the appropriate ``Stage`` from ``build_steps`` and
delegates to ``_run_stage``, which iterates through the steps, accumulates
output, and writes a per-stage log file after every step.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from . import display
from .build_steps import (
    Stage,
    build_stage,
    install_build_requirements_stage,
    install_dependencies_stage,
    prepare_tarball_stage,
)
from .config import BUILD_STEP_TIMEOUT_SECONDS, TOOL_EXEC_TIMEOUT_SECONDS, Config
from .lxd import ExecResult, LXDManager

log = logging.getLogger(__name__)

# Where the resolver stages the unified diff before invoking ``git apply``.
# Deliberately *not* under ``/tmp``: a freshly-copied validation container is
# still early in boot when apply_diff runs, and writes to ``/tmp`` can be
# shadowed by systemd's tmp mount/cleanup, leaving git with nothing to open
# ("can't open patch ... No such file or directory"). ``/root`` is on the
# container rootfs and stable from the moment the container starts.
_DIFF_PATH = "/root/resolver.diff"

_LOG_TAIL_LINES = 500
_ERROR_CONTEXT_BEFORE = 50
_ERROR_CONTEXT_AFTER = 50
_ERROR_TRAILING_TAIL = 100
# Word-boundary match on common failure markers, case-insensitive. Broad on
# purpose — false positives degrade to "model sees a useful chunk anyway" (no
# worse than the flat tail). Also covers C++ / CMake / dpkg-specific patterns
# that don't match the word-boundary form (e.g. "undefined reference").
# This is the *fallback* tier; _STRONG_ERROR_RE is preferred when it matches.
_ERROR_RE = re.compile(
    r"\b(error|failed|fatal)\b"
    r"|undefined reference"
    r"|cannot find"
    r"|CMake Error"
    r"|No such file"
    r"|ImportError",
    re.IGNORECASE,
)
# Structured, high-confidence failure markers — the lines that *terminate* a
# build rather than the benign "error"/"failed" tokens that litter a CMake
# configure phase ("-- Performing Test X - Failed", "Looking for ... - found").
# Anchoring on these lands the excerpt on the real cause instead of probe noise.
_STRONG_ERROR_RE = re.compile(
    r"^\s*CMake Error\b"               # message(FATAL_ERROR) / find_package failure
    r"|Configuring incomplete, errors occurred"
    r"|:\s*(?:fatal\s+)?error:"        # gcc/clang: file:line:col: error:
    r"|undefined reference to"         # linker
    r"|\*\*\*\s*\[[^]]*\]\s*Error\s*\d"  # make: *** [target] Error 1
    r"|\bmake(?:\[\d+\])?:\s*\*\*\*"   # make: *** stop
    r"|ninja:\s*build stopped"
    r"|recipe for target .* failed"
    r"|returned exit code [1-9]",      # dh_auto_* ... returned exit code N
    re.IGNORECASE,
)
# debuild runs with set -x, echoing all environment variables at the start.
# Some of those variable VALUES contain the word "error" (e.g.
# DH_OVERIDDEN_COMMAND=@echo 'error: ...'). Skip such assignment lines so
# the error scan finds real build failures, not env-dump noise.
_ENV_VAR_RE = re.compile(r"^[A-Z_][A-Z0-9_]*=")
# CMake status/progress lines (always prefixed with "-- ") are pure noise for
# error-finding: "-- Looking for X - found", "-- Performing Test Y - Failed".
# Real CMake errors are emitted as "CMake Error at ..." (no "-- " prefix), so
# excluding these never hides a genuine failure — it only stops a benign probe
# result from winning the first-match scan.
_BENIGN_LINE_RE = re.compile(r"^\s*--\s")


def _find_error_anchor(lines: list[str]) -> int | None:
    """Index of the line that best represents the build failure, or None.

    Two passes, both skipping env-var preamble and CMake ``-- `` status noise:
      1. The first *structured* failure marker (``_STRONG_ERROR_RE``) — the
         real cause for compile errors (first ``error:``) and CMake failures
         (the ``CMake Error at ...`` line, once probe noise is excluded).
      2. Failing that, the first broad ``_ERROR_RE`` match.
    Returning None means no marker was found at all.
    """

    def eligible(line: str) -> bool:
        return not (_ENV_VAR_RE.match(line) or _BENIGN_LINE_RE.match(line))

    for pattern in (_STRONG_ERROR_RE, _ERROR_RE):
        for i, line in enumerate(lines):
            if eligible(line) and pattern.search(line):
                return i
    return None


def first_error_line(log_tail: str) -> str:
    """Return the line that best represents the build failure.

    Prefers a structured marker (a real ``error:`` / ``CMake Error`` /
    make/linker failure) over the benign "error"/"failed" tokens that litter a
    CMake configure phase. Falls back to the last non-empty line if no marker
    is found.
    """
    lines = log_tail.splitlines()
    idx = _find_error_anchor(lines)
    if idx is not None:
        return lines[idx].strip()[:200]
    return log_tail.strip().splitlines()[-1][:200] if log_tail.strip() else ""


def _build_log_excerpt(full_output: str) -> str:
    """Extract a focused excerpt from build output.

    If an error marker is found (see ``_find_error_anchor``), return ±50 lines
    around it followed by the last 100 lines of the log (separated by a
    `--- snip ---` marker so the model can tell the sections apart). Otherwise
    return the flat 500-line tail.
    """
    lines = full_output.splitlines()
    first_hit = _find_error_anchor(lines)

    if first_hit is None:
        return "\n".join(lines[-_LOG_TAIL_LINES:])

    around_start = max(0, first_hit - _ERROR_CONTEXT_BEFORE)
    around_end = min(len(lines), first_hit + _ERROR_CONTEXT_AFTER + 1)
    around = lines[around_start:around_end]
    trailing_start = max(around_end, len(lines) - _ERROR_TRAILING_TAIL)
    trailing = lines[trailing_start:]

    parts = [
        f"--- first error context (lines {around_start + 1}-{around_end}) ---",
        "\n".join(around),
    ]
    if trailing and trailing_start > around_end:
        parts.extend(
            [
                f"--- snip ({trailing_start - around_end} lines) ---",
                f"--- trailing tail (lines {trailing_start + 1}-{len(lines)}) ---",
                "\n".join(trailing),
            ]
        )
    return "\n".join(parts)


@dataclass
class BuildOutcome:
    """Result of a build attempt with enough context to feed back to the model."""

    ok: bool
    stage: str
    returncode: int
    log_path: str  # path *inside the container* of the captured log
    log_tail: str  # last N lines, ready to splice into a prompt


class BuildRunner:
    def __init__(self, lxd: LXDManager, cfg: Config) -> None:
        self._lxd = lxd
        self._cfg = cfg

    @property
    def cfg(self) -> Config:
        return self._cfg

    @property
    def lxd(self) -> LXDManager:
        return self._lxd

    # ------------------------------------------------------------------
    # Build stages
    # ------------------------------------------------------------------

    def install_dependencies(self, container: str) -> ExecResult:
        display.stage_header("install_dependencies")
        return self._run_stage(container, install_dependencies_stage(self._cfg))

    def prepare_tarball(self, container: str) -> ExecResult:
        display.stage_header("prepare_tarball")
        return self._run_stage(container, prepare_tarball_stage(self._cfg))

    def install_build_requirements(self, container: str) -> ExecResult:
        display.stage_header("install_build_requirements")
        return self._run_stage(
            container, install_build_requirements_stage(self._cfg)
        )

    def build(self, container: str) -> BuildOutcome:
        display.stage_header("build")
        stage = build_stage(self._cfg)
        result = self._run_stage(container, stage)
        log_path = f"{self._cfg.container_log_dir}/{stage.name}.log"
        tail = _build_log_excerpt(result.stdout)
        outcome = BuildOutcome(
            ok=result.ok,
            stage=stage.name,
            returncode=result.returncode,
            log_path=log_path,
            log_tail=tail,
        )
        display.build_result(outcome.ok, outcome.stage, outcome.log_tail)
        return outcome

    # ------------------------------------------------------------------
    # Resolver-specific helpers
    # ------------------------------------------------------------------

    def patch_numstat(self, container: str, diff_text: str) -> ExecResult:
        """Dry-run ``git apply --numstat`` on a diff to learn the paths it touches.

        ``--numstat`` is parse-only (it never modifies the tree) and reports
        the post-``-p1`` target path for every file the diff would create,
        modify, delete, or rename-to. It is git's own path resolver, so the
        scope check and the real apply agree on what is touched -- unlike a
        hand-rolled header parser, which the prior ``_diff_target_paths``
        proved easy to slip past (``x/``-prefixed, header-only, or mode-only
        diffs yielded no targets while ``git apply`` still wrote the file).

        ``-z`` NUL-delimits records so paths with odd characters can't be
        misparsed. A non-zero exit means git could not parse the diff at all;
        the caller must refuse to apply it.
        """
        self._lxd.put_text(container, _DIFF_PATH, diff_text)
        return self._lxd.exec(
            container,
            ["git", "apply", "--numstat", "-z", _DIFF_PATH],
            cwd=self._cfg.container_workdir,
            check=False,
            timeout=TOOL_EXEC_TIMEOUT_SECONDS,
        )

    def apply_patch_to_tree(self, container: str, diff_text: str) -> ExecResult:
        """Apply a unified diff to the current working tree without resetting first.

        Unlike apply_diff, this preserves all uncommitted working-tree changes
        (patch files, rule edits, etc.) made earlier in the same session. Used
        by the model's apply_patch tool so mid-session state is not wiped.
        """
        self._lxd.put_text(container, _DIFF_PATH, diff_text)
        return self._lxd.exec(
            container,
            ["git", "apply", "--whitespace=nowarn", _DIFF_PATH],
            cwd=self._cfg.container_workdir,
        )

    def apply_diff(self, container: str, diff_text: str) -> ExecResult:
        """Apply a unified diff to a clean HEAD baseline inside the container.

        Used by validation to apply the model's final diff to a pristine
        container copy — the tree MUST be reset first so the diff applies
        against a known baseline. For mid-session applies (model's apply_patch
        tool), use apply_patch_to_tree instead to preserve prior changes.
        """
        self._lxd.put_text(container, _DIFF_PATH, diff_text)
        # Guard against the diff failing to materialise inside the container
        # (e.g. an empty capture, or a push that didn't land on a container
        # still mid-boot).  Surfacing this directly beats letting ``git apply``
        # report a cryptic "can't open patch" ENOENT downstream.
        check = self._lxd.exec_shell(
            container, f"test -s {_DIFF_PATH}", check=False
        )
        if not check.ok:
            return ExecResult(
                1,
                "",
                f"diff failed to materialise at {_DIFF_PATH} in {container} "
                "(empty capture or push did not land)",
            )
        # Reset working tree AND index to a clean HEAD baseline.
        # git checkout resets the working tree; git reset resets the index
        # (needed if apply_diff is called multiple times in the same container).
        self._lxd.exec_shell(
            container,
            f"cd {self._cfg.container_workdir} && "
            "git checkout HEAD -- . >/dev/null 2>&1; "
            "git reset HEAD -- . >/dev/null 2>&1",
            check=False,
        )
        # Remove untracked files in debian/ that the diff adds as new files.
        # git apply aborts (silently, exit 0) when a "new file" already exists
        # in the working tree, leaving every subsequent hunk unapplied.  This
        # happens when the pristine snapshot predates a fresh git clone (e.g.
        # patch files copied in before the resolver started).
        self._lxd.exec_shell(
            container,
            f"cd {self._cfg.container_workdir} && "
            "git ls-files -o debian/ | xargs -r rm -f",
            check=False,
        )
        return self._lxd.exec(
            container,
            [
                "git",
                "apply",
                "--index",
                "--whitespace=nowarn",
                _DIFF_PATH,
            ],
            cwd=self._cfg.container_workdir,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_stage(self, container: str, stage: Stage) -> ExecResult:
        log_path = f"{self._cfg.container_log_dir}/{stage.name}.log"
        buf = ""

        for step in stage.steps:
            display.step_start(step.argv)
            result = self._lxd.exec(
                container,
                step.argv,
                cwd=step.workdir,
                timeout=BUILD_STEP_TIMEOUT_SECONDS,
            )
            display.step_done(result.ok, result.returncode, result.stdout + result.stderr)
            buf += f"=== step: {step.argv} ===\n{result.stdout}"
            if result.stderr:
                buf += result.stderr + "\n"
            if result.timed_out:
                buf += (
                    f"=== step timed out after {BUILD_STEP_TIMEOUT_SECONDS}s "
                    "and was killed ===\n"
                )
            self._lxd.put_text(container, log_path, buf)
            if not result.ok and not step.allow_failure:
                return ExecResult(result.returncode, buf, result.stderr)

        return ExecResult(0, buf, "")
