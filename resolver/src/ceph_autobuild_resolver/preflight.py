"""Deterministic patch preflight: drop reversed patches, refresh drifted ones.

Two phases run before the model is ever invoked:

Phase 1 — Reversed patch detection:
  For each series entry, ``patch -F 0 -p1 --dry-run`` is run against a clean
  tree. "Reversed (or previously applied)" means the upstream already contains
  those changes; the entry and file are removed automatically.

Phase 2 — Drift auto-refresh:
  Surviving patches are walked with quilt. A patch that fails strict push
  (fuzz=0) but succeeds with fuzz=2 is correct in intent but has drifted line
  numbers. ``quilt refresh`` regenerates it with current context so dpkg-source
  will accept it without human intervention.
"""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass

from . import display
from .build_steps import TREE_RESET_SCRIPT
from .config import BUILD_STEP_TIMEOUT_SECONDS, TOOL_EXEC_TIMEOUT_SECONDS, Config
from .lxd import LXDError, LXDManager
from .transcript import Transcript

log = logging.getLogger(__name__)


@dataclass
class PreflightResult:
    """Outcome of the deterministic patch preflight pass.

    ``report`` is a human-readable per-patch status table passed to the model
    as part of its initial context.
    ``dropped`` lists patches removed from series + filesystem because their
    changes are already in the upstream source.
    ``refreshed`` lists patches whose hunk locations had drifted but were
    re-derived against the current upstream via quilt push -f + quilt refresh.
    """

    report: str
    dropped: list[str]
    refreshed: list[str]


def merge(first: PreflightResult, second: PreflightResult) -> PreflightResult:
    """Combine two preflight passes: latest report, union of changes.

    A second pass (after a failed post-preflight rebuild) starts from
    already-cleaned state, so its dropped/refreshed lists are typically
    empty. Keeping only the second result would erase the record of what
    preflight actually did from the PR summary; the report, by contrast,
    must reflect the *current* patch state, so the latest one wins.
    """

    def _union(a: list[str], b: list[str]) -> list[str]:
        return list(dict.fromkeys([*a, *b]))

    return PreflightResult(
        report=second.report,
        dropped=_union(first.dropped, second.dropped),
        refreshed=_union(first.refreshed, second.refreshed),
    )


def run(lxd: LXDManager, container: str, cfg: Config, series: str) -> PreflightResult:
    """Run the preflight; on any failure return an empty result.

    Best-effort: if the preflight crashes (LXD error, malformed series, etc.)
    we log and return an empty result so the resolver can still invoke the
    model loop with the unmodified patch state.
    """
    try:
        return _run_patch_preflight(lxd, container, cfg, series)
    except LXDError as exc:
        log.warning(
            "preflight aborted due to LXD error: %s — proceeding without it", exc
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "preflight aborted due to unexpected error: %s — proceeding without it",
            exc,
        )
    return PreflightResult(report="", dropped=[], refreshed=[])


def record(transcript: Transcript, result: PreflightResult) -> None:
    """Persist preflight outcome to the transcript and render to the terminal."""
    transcript.preflight(
        dropped=result.dropped,
        refreshed=result.refreshed,
        report=result.report,
    )
    display.preflight_summary(result.report, result.dropped, result.refreshed)


def format_summary(result: PreflightResult) -> str:
    """Human-readable description of what the preflight changed (used in the PR)."""
    parts: list[str] = []
    if result.dropped:
        parts.append(
            f"Removed {len(result.dropped)} patch(es) already merged into "
            f"the upstream source: " + ", ".join(result.dropped)
        )
    if result.refreshed:
        parts.append(
            f"Auto-refreshed {len(result.refreshed)} patch(es) whose "
            f"hunk locations had drifted (line numbers/context regenerated "
            f"via quilt refresh; semantic intent unchanged): "
            + ", ".join(result.refreshed)
        )
    return "\n\n".join(parts)


def _run_patch_preflight(
    lxd: LXDManager, container: str, cfg: Config, series: str
) -> PreflightResult:
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
            timeout=TOOL_EXEC_TIMEOUT_SECONDS,
        )
        combined = (result.stdout + result.stderr).strip()
        if not result.ok and (
            "Reversed" in combined or "previously applied" in combined
        ):
            log.info("preflight: %s — REVERSED, dropping from series", patch)
            _drop_patch(lxd, container, cfg, patch)
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
        rows: list[str] = []
        for p in patches:
            if p in dropped:
                rows.append(f"  {p}: REVERSED — auto-dropped from series")
            else:
                rows.append(f"  {p}: (phase 2 skipped — quilt unavailable)")
        return PreflightResult(report="\n".join(rows), dropped=dropped, refreshed=[])

    log.info(
        "preflight phase 2: walking %d surviving patches; auto-refreshing drift",
        len(surviving),
    )
    _reset_tree(lxd, container, workdir)

    # Quilt defaults to looking for ``patches/series``; Debian keeps it at
    # ``debian/patches/series``. Quilt normally learns the location from
    # ``.pc/.quilt_patches``, but _reset_tree just deleted ``.pc`` — without
    # this env every quilt call fails with "No series file found" and the
    # whole auto-refresh pass silently degrades to a no-op.
    quilt_env = {"QUILT_PATCHES": "debian/patches"}

    for patch in surviving:
        strict = lxd.exec(
            container,
            ["bash", "-c", "quilt push --fuzz=0"],
            cwd=workdir,
            env=quilt_env,
            check=False,
            timeout=TOOL_EXEC_TIMEOUT_SECONDS,
        )
        if strict.ok:
            statuses[patch] = "OK"
            log.debug("preflight: %s — OK", patch)
            continue
        forced = lxd.exec(
            container,
            ["bash", "-c", "quilt push -f --fuzz=2"],
            cwd=workdir,
            env=quilt_env,
            check=False,
            timeout=TOOL_EXEC_TIMEOUT_SECONDS,
        )
        if forced.ok:
            refresh = lxd.exec(
                container,
                ["bash", "-c", "quilt refresh"],
                cwd=workdir,
                env=quilt_env,
                check=False,
                timeout=TOOL_EXEC_TIMEOUT_SECONDS,
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
        err = _first_failure_line(forced.stdout + forced.stderr)
        log.warning("preflight: %s — FAIL: %s", patch, err)
        statuses[patch] = f"FAIL: {err}"
        break

    _reset_tree(lxd, container, workdir)

    rows = []
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


def _ensure_quilt(lxd: LXDManager, container: str) -> bool:
    have = lxd.exec(
        container, ["bash", "-c", "command -v quilt"], check=False,
        timeout=TOOL_EXEC_TIMEOUT_SECONDS,
    )
    if have.ok:
        return True
    log.info("preflight: quilt not present in container, installing")
    install = lxd.exec(
        container,
        ["bash", "-c", "sudo apt-get install -y quilt"],
        check=False,
        timeout=BUILD_STEP_TIMEOUT_SECONDS,
    )
    if not install.ok:
        log.warning(
            "preflight: failed to install quilt: %s",
            (install.stderr or install.stdout).strip()[:200],
        )
        return False
    verify = lxd.exec(
        container, ["bash", "-c", "command -v quilt"], check=False,
        timeout=TOOL_EXEC_TIMEOUT_SECONDS,
    )
    return verify.ok


def _reset_tree(lxd: LXDManager, container: str, workdir: str) -> None:
    """Pop quilt state and restore upstream files to their committed form."""
    lxd.exec(
        container, ["bash", "-c", TREE_RESET_SCRIPT], cwd=workdir, check=False,
        timeout=TOOL_EXEC_TIMEOUT_SECONDS,
    )


def _first_failure_line(output: str) -> str:
    for line in output.splitlines():
        low = line.lower()
        if "failed" in low or "malformed" in low or "error" in low:
            return line.strip()
    return output.strip()[:120]


def _drop_patch(
    lxd: LXDManager, container: str, cfg: Config, patch_name: str
) -> None:
    """Remove ``patch_name`` from series and delete the patch file."""
    workdir = cfg.container_workdir
    series_path = f"{workdir}/debian/patches/series"
    cur = lxd.exec(
        container, ["cat", series_path], check=False,
        timeout=TOOL_EXEC_TIMEOUT_SECONDS,
    )
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
        timeout=TOOL_EXEC_TIMEOUT_SECONDS,
    )
