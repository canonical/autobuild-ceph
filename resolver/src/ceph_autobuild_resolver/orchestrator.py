"""Top-level orchestrator: build → [resolution loop → validate] → output."""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass

from . import display, prompts, validation
from .budget import Budget
from .build_runner import BuildRunner
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
    preflight = _safe_preflight(lxd, container, cfg, patch_series)
    _record_preflight(transcript, preflight)
    current_failure = initial
    if preflight.dropped or preflight.refreshed:
        log.info(
            "preflight summary: dropped=%d refreshed=%d — retrying build",
            len(preflight.dropped), len(preflight.refreshed),
        )
        current_failure = runner.build(container)
        transcript.build_attempt(current_failure)
        if current_failure.ok:
            log.info("preflight auto-fix succeeded; skipping model loop")
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
                summary=_format_preflight_summary(preflight),
            )
        # Build still failing — refresh series + report for the model.
        patch_series = _capture_patch_series(lxd, container, cfg)
        preflight = _safe_preflight(lxd, container, cfg, patch_series)
        _record_preflight(transcript, preflight)

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
            cfg, file_tree, current_failure, patch_series, preflight.report
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

    if not loop_result.declared_resolved:
        transcript.outcome("loop_failed", stop_reason=loop_result.stop_reason)
        file_bug(
            BugPayload(
                matrix_name=matrix_name,
                failing_command=initial.stage,
                error_tail=initial.log_tail,
                transcript_path=str(transcript.path),
                stop_reason=loop_result.stop_reason,
            ),
            dry_run=dry_run_output,
        )
        return Outcome(1, f"resolution failed: {loop_result.stop_reason}")

    summary = loop_result.resolution_summary or "(no summary provided)"
    preflight_note = _format_preflight_summary(preflight)
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
    open_pr(
        PRPayload(
            matrix_name=matrix_name,
            summary=summary,
            diff=diff,
            failing_command=initial.stage,
            original_error_tail=initial.log_tail,
            transcript_path=str(transcript.path),
            flags={},
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


@dataclass
class PreflightResult:
    """Outcome of the deterministic patch preflight pass.

    ``report`` is a human-readable summary (passed into the model context if
    we end up invoking the loop).
    ``dropped`` lists patches removed from series + filesystem because their
    changes are already in the upstream source.
    ``refreshed`` lists patches whose hunk locations had drifted but were
    re-derived against the current upstream via quilt push -f + quilt refresh.
    """

    report: str
    dropped: list[str]
    refreshed: list[str]


def _run_patch_preflight(
    lxd: LXDManager, container: str, cfg: Config, series: str
) -> PreflightResult:
    """Two-phase deterministic audit of debian/patches/series.

    Phase 1 — drop reversed patches:
      For each series entry, run ``patch -F 0 -p1 --dry-run``. If the patch
      reports "Reversed (or previously applied)", the upstream already
      contains those changes; remove the entry and file.

    Phase 2 — auto-refresh drifted patches:
      Walk surviving patches with quilt. For each one, try ``quilt push
      --fuzz=0`` (matches dpkg-source's strictness). If it fails but
      ``quilt push -f --fuzz=2`` succeeds, the patch is correct in intent
      but the surrounding upstream context has shifted — ``quilt refresh``
      regenerates the .patch with current line numbers and context.

    Anything that survives both phases (and didn't break the series walk)
    is either OK or a genuine semantic conflict the model needs to address.
    """
    patches = [
        line.strip()
        for line in series.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not patches:
        return PreflightResult(report="", dropped=[], refreshed=[])

    workdir = cfg.container_workdir

    # ---- Phase 1: drop reversed patches ---------------------------------
    log.info("preflight phase 1: scanning %d patches for reversed state", len(patches))
    _reset_tree(lxd, container, workdir)
    dropped: list[str] = []
    for patch in patches:
        full = f"{workdir}/debian/patches/{patch}"
        result = lxd.exec(
            container,
            ["bash", "-c", f"patch -F 0 -p1 --dry-run < {shlex.quote(full)}"],
            cwd=workdir,
            check=False,
        )
        combined = (result.stdout + result.stderr).strip()
        if not result.ok and (
            "Reversed" in combined or "previously applied" in combined
        ):
            log.info("preflight: %s — REVERSED, dropping from series", patch)
            _drop_patch_in_container(lxd, container, cfg, patch)
            dropped.append(patch)

    # ---- Phase 2: walk surviving series with quilt; refresh drift -------
    surviving = [p for p in patches if p not in dropped]
    statuses: dict[str, str] = {}
    refreshed: list[str] = []

    if not _ensure_quilt(lxd, container):
        log.warning(
            "preflight phase 2: quilt unavailable in container; skipping "
            "auto-refresh. Patches with hunk drift will be left for the model."
        )
        # Build the report from phase 1 only and return.
        rows: list[str] = []
        for p in patches:
            if p in dropped:
                rows.append(f"  {p}: REVERSED — auto-dropped from series")
            else:
                rows.append(f"  {p}: (phase 2 skipped — quilt unavailable)")
        return PreflightResult(
            report="\n".join(rows), dropped=dropped, refreshed=[]
        )

    log.info(
        "preflight phase 2: walking %d surviving patches; auto-refreshing drift",
        len(surviving),
    )
    _reset_tree(lxd, container, workdir)

    for patch in surviving:
        # Strict push first (zero fuzz = same as dpkg-source's patch -F 0).
        strict = lxd.exec(
            container,
            ["bash", "-c", "quilt push --fuzz=0"],
            cwd=workdir,
            check=False,
        )
        if strict.ok:
            statuses[patch] = "OK"
            log.debug("preflight: %s — OK", patch)
            continue
        # Strict push failed. Try forcing with limited fuzz.
        forced = lxd.exec(
            container,
            ["bash", "-c", "quilt push -f --fuzz=2"],
            cwd=workdir,
            check=False,
        )
        if forced.ok:
            refresh = lxd.exec(
                container,
                ["bash", "-c", "quilt refresh"],
                cwd=workdir,
                check=False,
            )
            if refresh.ok:
                log.info(
                    "preflight: %s — drift detected, AUTO-REFRESHED via "
                    "quilt push -f + quilt refresh",
                    patch,
                )
                refreshed.append(patch)
                statuses[patch] = "AUTO-REFRESHED — line numbers updated"
                continue
            log.warning("preflight: %s — quilt refresh failed", patch)
            statuses[patch] = "FAIL: quilt refresh failed"
            break
        # Genuine conflict: even with fuzz the patch can't apply.
        err = _first_failure_line(forced.stdout + forced.stderr)
        log.warning("preflight: %s — FAIL: %s", patch, err)
        statuses[patch] = f"FAIL: {err}"
        break  # series walk halts; later patches can't be evaluated

    _reset_tree(lxd, container, workdir)

    # ---- Build report in original series order --------------------------
    rows: list[str] = []
    for p in patches:
        if p in dropped:
            rows.append(f"  {p}: REVERSED — auto-dropped from series")
        elif p in statuses:
            rows.append(f"  {p}: {statuses[p]}")
        else:
            rows.append(f"  {p}: not reached (earlier patch failed)")

    return PreflightResult(
        report="\n".join(rows),
        dropped=dropped,
        refreshed=refreshed,
    )


def _record_preflight(transcript: Transcript, preflight: PreflightResult) -> None:
    """Persist preflight outcome to the transcript and render to the terminal."""
    transcript.preflight(
        dropped=preflight.dropped,
        refreshed=preflight.refreshed,
        report=preflight.report,
    )
    display.preflight_summary(
        preflight.report, preflight.dropped, preflight.refreshed
    )


def _format_preflight_summary(preflight: PreflightResult) -> str:
    """Human-readable description of what the preflight changed."""
    parts: list[str] = []
    if preflight.dropped:
        parts.append(
            f"Removed {len(preflight.dropped)} patch(es) already merged into "
            f"the upstream source: " + ", ".join(preflight.dropped)
        )
    if preflight.refreshed:
        parts.append(
            f"Auto-refreshed {len(preflight.refreshed)} patch(es) whose "
            f"hunk locations had drifted (line numbers/context regenerated "
            f"via quilt refresh; semantic intent unchanged): "
            + ", ".join(preflight.refreshed)
        )
    return "\n\n".join(parts)


def _safe_preflight(
    lxd: LXDManager, container: str, cfg: Config, series: str
) -> PreflightResult:
    """Run the preflight; on any failure return an empty result.

    The preflight is a best-effort optimisation. If it crashes (LXD command
    not found, transient API error, malformed series file, etc.) we don't
    want to take down the whole resolver — the model can still do the work.
    """
    try:
        return _run_patch_preflight(lxd, container, cfg, series)
    except LXDError as exc:
        log.warning(
            "preflight aborted due to LXD error: %s — proceeding without it",
            exc,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "preflight aborted due to unexpected error: %s — proceeding without it",
            exc,
        )
    return PreflightResult(report="", dropped=[], refreshed=[])


def _ensure_quilt(lxd: LXDManager, container: str) -> bool:
    """Make quilt available in the container; return False if we can't."""
    have = lxd.exec(
        container, ["bash", "-c", "command -v quilt"], check=False
    )
    if have.ok:
        return True
    log.info("preflight: quilt not present in container, installing")
    install = lxd.exec(
        container,
        ["bash", "-c", "sudo apt-get install -y quilt"],
        check=False,
    )
    if not install.ok:
        log.warning(
            "preflight: failed to install quilt: %s",
            (install.stderr or install.stdout).strip()[:200],
        )
        return False
    verify = lxd.exec(
        container, ["bash", "-c", "command -v quilt"], check=False
    )
    return verify.ok


def _reset_tree(lxd: LXDManager, container: str, workdir: str) -> None:
    """Pop quilt state and reset upstream files to their committed form.

    Mirrors the reset performed at the start of build_stage so the preflight
    sees the same clean tree dpkg-source would.
    """
    lxd.exec(
        container,
        [
            "bash",
            "-c",
            "quilt pop -a 2>/dev/null || true; "
            "rm -rf .pc; "
            "git checkout HEAD -- . ':(exclude)debian' 2>/dev/null || true; "
            "git clean -fd -e debian/ 2>/dev/null || true; "
            "true",
        ],
        cwd=workdir,
        check=False,
    )


def _first_failure_line(output: str) -> str:
    for line in output.splitlines():
        low = line.lower()
        if "failed" in low or "malformed" in low or "error" in low:
            return line.strip()
    return output.strip()[:120]


def _drop_patch_in_container(
    lxd: LXDManager, container: str, cfg: Config, patch_name: str
) -> None:
    """Remove ``patch_name`` from series and delete the patch file."""
    workdir = cfg.container_workdir
    series_path = f"{workdir}/debian/patches/series"
    cur = lxd.exec(container, ["cat", series_path], check=False)
    if cur.ok:
        kept = [
            line for line in cur.stdout.splitlines() if line.strip() != patch_name
        ]
        new_series = "\n".join(kept)
        if new_series and not new_series.endswith("\n"):
            new_series += "\n"
        lxd.put_text(container, series_path, new_series)
    lxd.exec(
        container,
        ["rm", "-f", f"{workdir}/debian/patches/{patch_name}"],
        check=False,
    )


def _capture_diff(lxd: LXDManager, container: str, cfg: Config) -> str:
    """Capture the model's accumulated changes as a unified diff against HEAD.

    Scope git-add to model-writable paths only (debian/patches/ + debian root
    files).  git add -A would also stage debhelper build artifacts under
    debian/<pkg>/ and quilt-created .orig files in upstream src/, both of which
    make git apply --index fail during validation.
    """
    # Stage only the paths the model is allowed to change.  debhelper staging
    # trees (debian/ceph-*/…) and quilt artefacts in src/ are intentionally
    # excluded.
    stage_cmd = (
        f"cd {cfg.container_workdir} && "
        "git reset HEAD -- . 2>/dev/null; "  # unstage any prior git add -A residue
        "git add -- debian/patches/ debian/rules debian/control "
        "debian/changelog debian/copyright debian/compat 2>/dev/null; "
        "git diff --staged"
    )
    result = lxd.exec_shell(container, stage_cmd, check=False)
    return result.stdout
