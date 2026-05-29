"""Tests for the pristine-prep flow."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from ceph_autobuild_resolver import prep
from ceph_autobuild_resolver.lxd import ExecResult, LXDError


@dataclass
class StubLXD:
    """Tracks operations and lets tests script ``exec`` outcomes."""

    container_exists: bool = False
    snapshot_exists: bool = False
    launched: list[tuple[str, str]] = field(default_factory=list)
    snapshots_created: list[tuple[str, str]] = field(default_factory=list)
    snapshots_deleted: list[tuple[str, str]] = field(default_factory=list)
    execs: list[list[str]] = field(default_factory=list)
    exec_returncode: int = 0

    def launch(self, image: str, name: str) -> None:
        self.launched.append((image, name))
        self.container_exists = True

    def snapshot(self, name: str, snapshot: str) -> None:
        self.snapshots_created.append((name, snapshot))
        self.snapshot_exists = True

    def exec(self, container, argv, *, env=None, check=False, cwd=None) -> ExecResult:
        self.execs.append(list(argv))
        return ExecResult(self.exec_returncode, "", "")

    def exec_shell(self, container, script, *, env=None, check=False, cwd=None) -> ExecResult:
        self.execs.append(["bash", "-c", script])
        return ExecResult(self.exec_returncode, "", "")

    def put_text(self, container, remote_path, content) -> None:
        pass

    def _instance(self, name):
        if not self.container_exists:
            raise LXDError(f"no such container: {name}")
        outer = self

        class _Snapshots:
            def get(self, snap_name):
                if not outer.snapshot_exists:
                    raise RuntimeError("not found")
                class _Snap:
                    def delete(self, wait=True):
                        outer.snapshots_deleted.append((name, snap_name))
                        outer.snapshot_exists = False
                return _Snap()

        class _Inst:
            snapshots = _Snapshots()

        return _Inst()


def test_prep_launches_when_missing(cfg):
    lxd = StubLXD(container_exists=False)

    out = prep.run(
        cfg=cfg,
        image="ubuntu:24.04",
        container="ceph-build",
        snapshot="pristine",
        lxd=lxd,
    )

    assert out.container == "ceph-build"
    assert lxd.launched == [("ubuntu:24.04", "ceph-build")]
    assert lxd.snapshots_created == [("ceph-build", "pristine")]
    # Verify prep stages were invoked by checking for characteristic commands.
    all_argv = [e for e in lxd.execs]
    flat = " ".join(" ".join(a) for a in all_argv)
    assert "apt" in flat
    assert "git" in flat


def test_prep_skips_launch_when_existing(cfg):
    lxd = StubLXD(container_exists=True)

    prep.run(
        cfg=cfg,
        image="ubuntu:24.04",
        container="ceph-build",
        snapshot="pristine",
        lxd=lxd,
        skip_existing=True,
    )
    assert lxd.launched == []
    assert lxd.snapshots_created == [("ceph-build", "pristine")]


def test_prep_overwrites_existing_snapshot(cfg):
    lxd = StubLXD(container_exists=True, snapshot_exists=True)

    prep.run(
        cfg=cfg,
        image="ubuntu:24.04",
        container="ceph-build",
        snapshot="pristine",
        lxd=lxd,
    )
    assert lxd.snapshots_deleted == [("ceph-build", "pristine")]
    assert lxd.snapshots_created == [("ceph-build", "pristine")]


def test_prep_propagates_stage_failure(cfg):
    lxd = StubLXD(container_exists=True, exec_returncode=1)

    with pytest.raises(LXDError):
        prep.run(
            cfg=cfg,
            image="ubuntu:24.04",
            container="ceph-build",
            snapshot="pristine",
            lxd=lxd,
        )
