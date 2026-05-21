"""Pristine container prep.

Reproduces what the GitHub Actions workflow does as separate steps today,
collapsed into one Python entry point so the resolver can be exercised
end-to-end in a single command (locally, in a VM, or in CI).

The result is a container in a clean post-prep state plus a snapshot named
by the caller (``pristine`` by default). Subsequent resolver runs copy from
that snapshot for both iteration containers and the validation rebuild.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .build_runner import BuildRunner
from .config import Config
from .lxd import LXDError, LXDManager

log = logging.getLogger(__name__)


@dataclass
class PrepOutcome:
    container: str
    snapshot: str
    image: str


def run(
    *,
    cfg: Config,
    image: str,
    container: str,
    snapshot: str,
    skip_existing: bool = True,
    lxd: LXDManager | None = None,
) -> PrepOutcome:
    """Launch ``container`` from ``image``, run prep stages, and snapshot.

    If ``skip_existing`` is True (the default) and the container already
    exists, the existing one is reused — useful for re-running prep after
    fixing a transient failure mid-prep without paying the launch cost again.
    Snapshot creation is also idempotent: re-snapshotting overwrites.
    """
    lxd = lxd or LXDManager()
    runner = BuildRunner(lxd, cfg)

    if not skip_existing or not _container_exists(lxd, container):
        log.info("launching %s from %s", container, image)
        lxd.launch(image, container)
    else:
        log.info("reusing existing container %s", container)

    log.info("install_dependencies")
    res = runner.install_dependencies(container)
    if not res.ok:
        raise LXDError(f"install_dependencies failed:\n{res.stderr}")

    log.info("prepare_tarball")
    res = runner.prepare_tarball(container)
    if not res.ok:
        raise LXDError(f"prepare_tarball failed:\n{res.stderr}")

    log.info("install_build_requirements")
    res = runner.install_build_requirements(container)
    if not res.ok:
        raise LXDError(f"install_build_requirements failed:\n{res.stderr}")

    log.info("snapshotting %s -> %s", container, snapshot)
    # Best-effort delete of an old snapshot of the same name; pylxd raises if
    # we try to create one that already exists.
    _delete_snapshot_if_exists(lxd, container, snapshot)
    lxd.snapshot(container, snapshot)

    return PrepOutcome(container=container, snapshot=snapshot, image=image)


def _container_exists(lxd: LXDManager, name: str) -> bool:
    try:
        lxd._instance(name)  # noqa: SLF001 — internal helper, fine for our use
        return True
    except LXDError:
        return False


def _delete_snapshot_if_exists(lxd: LXDManager, container: str, snapshot: str) -> None:
    try:
        inst = lxd._instance(container)  # noqa: SLF001
    except LXDError:
        return
    try:
        existing = inst.snapshots.get(snapshot)
    except Exception:  # noqa: BLE001 — pylxd raises NotFound; we don't import it
        return
    try:
        existing.delete(wait=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not delete existing snapshot %s: %s", snapshot, exc)
