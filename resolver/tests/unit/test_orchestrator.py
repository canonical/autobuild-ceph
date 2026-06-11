"""Tests for orchestrator helpers (diff capture staging)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ceph_autobuild_resolver.orchestrator import _diff_stage_script

_CONTROL = """\
Source: ceph
Maintainer: nobody

Package: ceph-common
Architecture: any

Package: librados2
Architecture: any
"""


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "PATH": "/usr/bin:/bin",
        },
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "ceph"
    (repo / "debian/patches").mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / "debian/control").write_text(_CONTROL)
    (repo / "debian/rules").write_text("#!/usr/bin/make -f\n")
    (repo / "debian/ceph-common.install").write_text("usr/bin/ceph\n")
    (repo / "debian/patches/series").write_text("old.patch\n")
    (repo / "debian/patches/old.patch").write_text("--- a/x\n+++ b/x\n")
    (repo / "src/foo.cc").write_text("int x;\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    return repo


def _run_stage_script(repo: Path) -> str:
    return subprocess.check_output(
        ["bash", "-c", _diff_stage_script(str(repo))], text=True
    )


def test_diff_captures_all_model_edits_under_debian(repo: Path):
    # Edits the prompt explicitly directs: control, an .install file, and a
    # dropped patch -- the old six-file allowlist captured only some of these.
    (repo / "debian/control").write_text(_CONTROL.replace("librados2", "librados3"))
    (repo / "debian/ceph-common.install").write_text("usr/bin/ceph\nusr/bin/rbd\n")
    (repo / "debian/new-pkg.install").write_text("usr/lib/new\n")
    (repo / "debian/patches/old.patch").unlink()
    (repo / "debian/patches/series").write_text("")

    diff = _run_stage_script(repo)

    assert "librados3" in diff
    assert "usr/bin/rbd" in diff
    assert "debian/new-pkg.install" in diff
    assert "deleted file" in diff and "old.patch" in diff


def test_diff_excludes_build_artifacts(repo: Path):
    (repo / "debian/rules").write_text("#!/usr/bin/make -f\n# changed\n")
    # debhelper staging trees are named after binary packages in control.
    (repo / "debian/ceph-common/usr/bin").mkdir(parents=True)
    (repo / "debian/ceph-common/usr/bin/ceph").write_text("ELF")
    (repo / "debian/librados2").mkdir()
    (repo / "debian/librados2/x").write_text("ELF")
    (repo / "debian/.debhelper").mkdir()
    (repo / "debian/.debhelper/log").write_text("x")
    (repo / "debian/tmp").mkdir()
    (repo / "debian/tmp/y").write_text("x")
    (repo / "debian/files").write_text("ceph_1.deb misc optional\n")
    (repo / "debian/ceph-common.substvars").write_text("misc:Depends=\n")
    (repo / "debian/ceph-common.debhelper.log").write_text("x")
    (repo / "debian/debhelper-build-stamp").write_text("x")
    (repo / "debian/rules.orig").write_text("old\n")
    (repo / "src/foo.cc.orig").write_text("int y;\n")

    diff = _run_stage_script(repo)

    assert "# changed" in diff
    for forbidden in (
        "ceph-common/usr/bin",
        "librados2/x",
        ".debhelper",
        "debian/tmp",
        "debian/files",
        "substvars",
        "build-stamp",
        ".orig",
    ):
        assert forbidden not in diff, forbidden


def test_diff_still_captures_patches_dir(repo: Path):
    (repo / "debian/patches/fix.patch").write_text("--- a/y\n+++ b/y\n")
    (repo / "debian/patches/series").write_text("old.patch\nfix.patch\n")

    diff = _run_stage_script(repo)

    assert "debian/patches/fix.patch" in diff
    assert "fix.patch" in diff
