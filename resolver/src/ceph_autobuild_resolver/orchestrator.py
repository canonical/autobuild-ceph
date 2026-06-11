"""Top-level orchestrator: build → [resolution loop → validate] → output."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from . import display, preflight as pf, prompts, validation
from .budget import Budget
from .build_runner import BuildRunner, first_error_line
from .build_steps import CONTAINER_CCACHE_DIR
from .config import Config
from .loop import run_loop
from .lxd import LXDError, LXDManager
from .output.launchpad_bug import BugPayload, file_bug
from .output.launchpad_pr import PRPayload, open_pr
from .providers.factory import build as build_provider
from .tools.dispatch import Dispatcher
from .tools.execution import ExecutionHandlers
from .tools.filesystem import FilesystemHandlers
from .tools.schema import all_tools
from .tools.search import SearchHandlers
from .transcript import Transcript

log = logging.getLogger(__name__)


@dataclass
class Outcome:
    exit_code: int
    description: str


def run(
    *,
    cfg: Config,
    container: str,
    pristine_snapshot: str,
    matrix_name: str,
    transcript_path: str,
    dry_run_output: bool,
) -> Outcome:
    display.startup_info(cfg.model_name, cfg.provider, cfg.ccache_host_dir)
    _start = time.monotonic()
    transcript = Transcript(transcript_path)
    lxd = LXDManager()
    runner = BuildRunner(lxd, cfg)

    # 0. Restore the iteration container to its post-prep snapshot. The
    #    container is reused across resolve invocations and accumulates state
    #    — partially applied quilt patches, leftover .pc/, modified upstream
    #    files our git-checkout reset can miss. Restoring from snapshot
    #    guarantees every resolve starts from the same clean state without
    #    paying the cost of a full re-prep (clone + tarball + apt installs).
    #    Best-effort: if the snapshot is missing we log and continue.
    try:
        log.info(
            "restoring %s to snapshot %s for clean start",
            container, pristine_snapshot,
        )
        lxd.restore_snapshot(container, pristine_snapshot)
    except LXDError as exc:
        log.warning(
            "could not restore %s/%s: %s — proceeding with current "
            "container state (may have stale modifications from prior runs)",
            container, pristine_snapshot, exc,
        )

    if cfg.ccache_host_dir:
        from .prep import _prepare_ccache_dir
        if _prepare_ccache_dir(cfg.ccache_host_dir):
            log.info("ensuring ccache device %s -> %s", cfg.ccache_host_dir, CONTAINER_CCACHE_DIR)
            lxd.attach_disk_device(container, "ccache", cfg.ccache_host_dir, CONTAINER_CCACHE_DIR)

    # 1. First build attempt against the pristine state. If it works, we're
    #    done before touching the model.
    initial = runner.build(container)
    transcript.build_attempt(initial)

    if initial.ok:
        log.info("initial build succeeded; resolver no-op")
        return Outcome(0, "no resolution needed")

    # 2. Deterministic auto-fix pass: drop any patches whose changes are
    #    already in upstream. This is a mechanical fact, not a decision —
    #    no need to involve the model. Many CI failures are just "patch X
    #    got merged upstream in this release" and resolve here.
    patch_series = _capture_patch_series(lxd, container, cfg)
    pf_result = pf.run(lxd, container, cfg, patch_series)
    pf.record(transcript, pf_result)
    current_failure = initial
    if pf_result.dropped or pf_result.refreshed:
        log.info(
            "preflight summary: dropped=%d refreshed=%d — retrying build",
            len(pf_result.dropped), len(pf_result.refreshed),
        )
        current_failure = runner.build(container)
        transcript.build_attempt(current_failure)
        if current_failure.ok:
            log.info("preflight auto-fix succeeded; skipping model loop")
            # No emit_summary here: we don't have a budget object yet since
            # we never entered the loop. The run summary is skipped for the
            # preflight-success fast path; the PR payload carries the story.
            return _validate_and_publish(
                lxd=lxd,
                runner=runner,
                cfg=cfg,
                container=container,
                pristine_snapshot=pristine_snapshot,
                matrix_name=matrix_name,
                transcript=transcript,
                dry_run_output=dry_run_output,
                initial=initial,
                summary=pf.format_summary(pf_result),
            )
        # Build still failing — refresh series + report for the model.
        patch_series = _capture_patch_series(lxd, container, cfg)
        pf_result = pf.run(lxd, container, cfg, patch_series)
        pf.record(transcript, pf_result)

    # 3. Set up the resolution loop with the (possibly cleaned) state.
    provider = build_provider(cfg)
    provider.declare_tools(all_tools())

    fs_handlers = FilesystemHandlers(
        lxd=lxd, cfg=cfg, container=container, runner=runner
    )
    search_handlers = SearchHandlers(lxd=lxd, cfg=cfg, container=container)
    exec_handlers = ExecutionHandlers(runner=runner, container=container)
    dispatcher = Dispatcher(
        filesystem=fs_handlers,
        search=search_handlers,
        execution=exec_handlers,
        build_log_path=current_failure.log_path,
    )

    file_tree = _capture_file_tree(lxd, container, cfg)
    history = [
        prompts.system_message(),
        prompts.initial_user_message(
            cfg, file_tree, current_failure, patch_series, pf_result.report
        ),
    ]
    transcript.initial_context(
        {
            "stage": current_failure.stage,
            "returncode": current_failure.returncode,
            "log_path": current_failure.log_path,
        }
    )

    budget = Budget.from_config(cfg)

    loop_result = run_loop(
        history=history,
        provider=provider,
        dispatcher=dispatcher,
        budget=budget,
        transcript=transcript,
    )

    def _emit_summary(*, success, stop_reason, resolution_summary="", diff="", last_error=""):
        display.run_summary(
            success=success,
            stop_reason=stop_reason,
            iterations=budget.iterations_used,
            total_tokens=budget.total_tokens_used,
            elapsed_seconds=time.monotonic() - _start,
            initial_error=first_error_line(current_failure.log_tail),
            resolution_summary=resolution_summary or "",
            diff=diff,
            last_build_error=last_error,
        )

    if not loop_result.declared_resolved:
        transcript.outcome("loop_failed", stop_reason=loop_result.stop_reason)
        last_error = _last_build_error(loop_result.history)
        _emit_summary(
            success=False,
            stop_reason=loop_result.stop_reason,
            last_error=last_error,
        )
        # For declare_unresolvable, the model's explanation is more useful
        # than the raw build log tail. Use it as the error_tail so it appears
        # in the CI failure output and the eventual Launchpad bug description.
        error_tail = (
            loop_result.resolution_summary or initial.log_tail
            if loop_result.stop_reason == "declared_unresolvable"
            else initial.log_tail
        )
        file_bug(
            BugPayload(
                matrix_name=matrix_name,
                failing_command=initial.stage,
                error_tail=error_tail,
                transcript_path=str(transcript.path),
                stop_reason=loop_result.stop_reason,
            ),
            dry_run=dry_run_output,
        )
        return Outcome(1, f"resolution failed: {loop_result.stop_reason}")

    summary = loop_result.resolution_summary or "(no summary provided)"
    preflight_note = pf.format_summary(pf_result)
    if preflight_note:
        summary = summary + "\n\n" + preflight_note
    return _validate_and_publish(
        lxd=lxd,
        runner=runner,
        cfg=cfg,
        container=container,
        pristine_snapshot=pristine_snapshot,
        matrix_name=matrix_name,
        transcript=transcript,
        dry_run_output=dry_run_output,
        initial=initial,
        summary=summary,
        emit_summary=_emit_summary,
    )


def _validate_and_publish(
    *,
    lxd: LXDManager,
    runner: BuildRunner,
    cfg: Config,
    container: str,
    pristine_snapshot: str,
    matrix_name: str,
    transcript: Transcript,
    dry_run_output: bool,
    initial,  # BuildOutcome
    summary: str,
    emit_summary=None,
) -> Outcome:
    """Validate via clean rebuild and either open a PR or file a bug."""
    diff = _capture_diff(lxd, container, cfg)
    val = validation.validate(
        lxd=lxd,
        runner=runner,
        pristine_container=container,
        pristine_snapshot=pristine_snapshot,
        diff=diff,
    )
    transcript.validation(
        ok=val.ok,
        stage=val.build_outcome.stage,
        returncode=val.build_outcome.returncode,
    )

    if not val.ok:
        transcript.outcome("validation_failed")
        if emit_summary:
            emit_summary(
                success=False,
                stop_reason="validation_failed",
                diff=diff,
                last_error=first_error_line(
                    val.build_outcome.log_tail or val.apply_stderr
                ),
            )
        file_bug(
            BugPayload(
                matrix_name=matrix_name,
                failing_command=val.build_outcome.stage,
                error_tail=val.build_outcome.log_tail or val.apply_stderr,
                transcript_path=str(transcript.path),
                stop_reason="validation_failed",
            ),
            dry_run=dry_run_output,
        )
        return Outcome(1, "validation failed on clean rebuild")

    transcript.outcome("success")
    if emit_summary:
        emit_summary(success=True, stop_reason="resolved", resolution_summary=summary, diff=diff)
    open_pr(
        PRPayload(
            matrix_name=matrix_name,
            summary=summary,
            diff=diff,
            transcript_path=str(transcript.path),
        ),
        dry_run=dry_run_output,
    )
    return Outcome(0, "resolved and validated")


def _capture_file_tree(lxd: LXDManager, container: str, cfg: Config) -> str:
    """Snapshot the working tree's top-level structure for the initial context."""
    # ``find ... -maxdepth 2`` keeps the output bounded; the model can
    # explore deeper via grep / read_files.
    result = lxd.exec(
        container,
        [
            "find",
            cfg.container_workdir,
            "-maxdepth",
            "2",
            "-not",
            "-path",
            "*/.git/*",
        ],
        check=False,
    )
    return result.stdout


def _capture_patch_series(lxd: LXDManager, container: str, cfg: Config) -> str:
    result = lxd.exec(
        container,
        ["cat", f"{cfg.container_workdir}/debian/patches/series"],
        check=False,
    )
    return result.stdout if result.ok else ""



# Build artifacts under debian/ that must never enter the captured diff:
# they would make git apply --index fail during validation. Everything else
# under debian/ is model-writable (the prompt directs edits to e.g.
# debian/*.install) and must be captured.
_DIFF_EXCLUDES = (
    ":!debian/.debhelper",
    ":!debian/tmp",
    ":!debian/files",
    ":!debian/*.substvars",
    ":!debian/*.debhelper.log",
    ":!debian/*.debhelper",
    ":!debian/debhelper-build-stamp",
    ":!debian/*.orig",
)


def _diff_stage_script(workdir: str) -> str:
    """Shell script that stages every model change under debian/ and prints it.

    An excludelist, not an allowlist: the prompt tells the model to edit
    arbitrary files under debian/ (control, *.install, rules, ...), so an
    allowlist silently drops directed edits from the captured diff -- the
    published diff then no longer matches what the model actually tested.

    Excluded are the debhelper/quilt build artifacts only: the fixed names
    above plus per-package staging trees, whose directory names are exactly
    the binary package names declared in debian/control.  ':!...' exclude
    pathspecs don't trigger git add's "pathspec did not match" fatal (that
    applies to positive pathspecs only, and debian/ always matches), and
    'git add -- debian/' also stages deletions of tracked files.  Quilt
    .orig artifacts in upstream src/ are never staged because the only
    positive pathspec is debian/.
    """
    excludes = " ".join(f'"{e}"' for e in _DIFF_EXCLUDES)
    return (
        f"cd {workdir} && "
        "git reset HEAD -- . >/dev/null 2>&1; "  # git reset writes status to stdout
        "ex=''; "
        "for p in $(awk '/^Package:/{print $2}' debian/control 2>/dev/null); do "
        '  ex="$ex :!debian/$p"; '
        "done; "
        f"git add -- debian/ {excludes} $ex; "
        "git diff --staged"
    )


def _capture_diff(lxd: LXDManager, container: str, cfg: Config) -> str:
    """Capture the model's accumulated changes as a unified diff against HEAD."""
    result = lxd.exec_shell(container, _diff_stage_script(cfg.container_workdir), check=False)
    return result.stdout


def _last_build_error(history) -> str:
    """Scan loop history for the first error line from the last failed run_build."""
    last = ""
    for msg in history:
        if msg.role != "tool":
            continue
        for r in msg.tool_results:
            if r.name == "run_build" and not r.payload.get("ok"):
                tail = r.payload.get("log_tail") or ""
                line = first_error_line(tail)
                if line:
                    last = line
    return last
