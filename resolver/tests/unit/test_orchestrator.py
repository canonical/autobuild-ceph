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


def test_captured_diff_reapplies_on_clean_tree(repo: Path):
    """The outcome-integrity contract: the captured diff must re-apply via
    `git apply --index` on a clean tree -- exactly what validation.apply_diff
    does on a pristine container. A regression in the staging script (e.g. a
    leaked build artifact) would make validation fail a genuinely good fix."""
    # Model edits across the debian/ surface, plus a debhelper artifact that
    # must NOT enter the diff.
    (repo / "debian/control").write_text(_CONTROL.replace("librados2", "librados3"))
    (repo / "debian/ceph-common.install").write_text("usr/bin/ceph\nusr/bin/rbd\n")
    (repo / "debian/patches/new.patch").write_text("--- a/y\n+++ b/y\n")
    (repo / "debian/patches/series").write_text("old.patch\nnew.patch\n")
    (repo / "debian/ceph-common").mkdir()
    (repo / "debian/ceph-common/usr").mkdir()
    (repo / "debian/ceph-common/usr/file").write_text("BUILT ARTIFACT")

    diff = _run_stage_script(repo)

    # Reset to a clean HEAD tree (what the pristine validation container is).
    _git(repo, "reset", "-q", "--hard")
    _git(repo, "clean", "-fdq")

    # Apply exactly as validation does.
    (repo / "patch.diff").write_text(diff)
    proc = subprocess.run(
        ["git", "-C", str(repo), "apply", "--index", "--whitespace=nowarn", "patch.diff"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"captured diff did not re-apply: {proc.stderr}"

    # The model's edits are present after re-apply; the artifact is not.
    assert "librados3" in (repo / "debian/control").read_text()
    assert "usr/bin/rbd" in (repo / "debian/ceph-common.install").read_text()
    assert (repo / "debian/patches/new.patch").exists()
    assert not (repo / "debian/ceph-common/usr/file").exists()


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


# ---------------------------------------------------------------------------
# Failure-path helpers
# ---------------------------------------------------------------------------

from ceph_autobuild_resolver.orchestrator import _last_failed_build_tail
from ceph_autobuild_resolver.preflight import PreflightResult, merge
from ceph_autobuild_resolver.providers.base import Message, ToolResult


def _build_msg(ok: bool, tail: str, skipped: bool = False) -> Message:
    payload = {"ok": ok, "log_tail": tail}
    if skipped:
        payload["skipped"] = True
    return Message(
        role="tool",
        tool_results=[ToolResult(call_id="c", name="run_build", payload=payload)],
    )


def test_last_failed_build_tail_prefers_latest_failure():
    history = [
        _build_msg(False, "error: initial failure"),
        _build_msg(False, "error: later, different failure"),
        _build_msg(True, "ok"),
    ]
    assert _last_failed_build_tail(history) == "error: later, different failure"


def test_last_failed_build_tail_skips_refused_builds():
    history = [
        _build_msg(False, "error: real failure"),
        _build_msg(False, "Refusing to re-run the build", skipped=True),
    ]
    assert _last_failed_build_tail(history) == "error: real failure"


def test_last_failed_build_tail_empty_when_no_failures():
    assert _last_failed_build_tail([_build_msg(True, "ok")]) == ""


def test_preflight_merge_keeps_first_pass_changes_and_latest_report():
    first = PreflightResult(report="r1", dropped=["a.patch"], refreshed=["b.patch"])
    second = PreflightResult(report="r2", dropped=[], refreshed=["b.patch"])
    merged = merge(first, second)
    assert merged.report == "r2"
    assert merged.dropped == ["a.patch"]
    assert merged.refreshed == ["b.patch"]


def test_failure_error_tail_prefers_authoritative_last_build():
    """History compression can evict the last build's tool result; the
    BuildOutcome held by the execution handlers must win over the history
    scan and the pre-loop fallback (CI run 27377122980 reported the initial
    dpkg-source error while the real blocker was a link failure)."""
    from ceph_autobuild_resolver.build_runner import BuildOutcome
    from ceph_autobuild_resolver.orchestrator import _failure_error_tail

    last = BuildOutcome(
        ok=False,
        stage="build",
        returncode=2,
        log_path="/root/build-logs/build.log",
        log_tail="error: undefined reference to get_rgw_options()",
    )
    # Compressed-away history: no run_build results survive.
    tail = _failure_error_tail(last, [], "dpkg-source: error: initial failure")
    assert "get_rgw_options" in tail


def test_failure_error_tail_falls_back_to_history_then_initial():
    from ceph_autobuild_resolver.build_runner import BuildOutcome
    from ceph_autobuild_resolver.orchestrator import _failure_error_tail

    ok_build = BuildOutcome(
        ok=True, stage="build", returncode=0, log_path="x", log_tail="ok"
    )
    history = [_build_msg(False, "error: from history")]
    assert _failure_error_tail(ok_build, history, "initial") == "error: from history"
    assert _failure_error_tail(ok_build, [], "initial") == "initial"
    assert _failure_error_tail(None, [], "initial") == "initial"


# ---------------------------------------------------------------------------
# _validate_and_publish: infra-fault must not discard a resolved fix
# ---------------------------------------------------------------------------

def test_validation_lxd_fault_persists_diff_and_does_not_crash(monkeypatch, tmp_path, cfg):
    """An LXDError during validation (snapshot copy/exec) must not propagate
    as a generic crash: the captured diff is written to CI_DIFF_FILE before
    validation runs, and the run returns a distinct validation_error outcome."""
    from ceph_autobuild_resolver import orchestrator
    from ceph_autobuild_resolver.build_runner import BuildOutcome
    from ceph_autobuild_resolver.lxd import ExecResult, LXDError
    from ceph_autobuild_resolver.transcript import Transcript

    captured_diff = "diff --git a/debian/rules b/debian/rules\n+x\n"

    class FaultyLXD:
        def exec_shell(self, *a, **k):
            return ExecResult(0, captured_diff, "")
        def exec(self, *a, **k):
            return ExecResult(0, "", "")
        def delete(self, *a, **k):
            pass
        def copy_from_snapshot(self, *a, **k):
            raise LXDError("transient: could not copy snapshot")

    diff_file = tmp_path / "resolver-diff.patch"
    monkeypatch.setenv("CI_DIFF_FILE", str(diff_file))

    outcome = orchestrator._validate_and_publish(
        lxd=FaultyLXD(),
        runner=object(),
        cfg=cfg,
        container="ceph-build",
        pristine_snapshot="pristine",
        matrix_name="resolute",
        transcript=Transcript(tmp_path / "t.jsonl"),
        dry_run_output=True,
        summary="fixed it",
    )

    assert outcome.exit_code == 1
    # The fix survived the infra fault.
    assert diff_file.read_text(encoding="utf-8") == captured_diff
    # Recorded as validation_error, not a bare crash.
    transcript_text = (tmp_path / "t.jsonl").read_text()
    assert "validation_error" in transcript_text
