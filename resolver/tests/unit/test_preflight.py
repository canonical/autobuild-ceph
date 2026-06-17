"""Preflight phase 2: quilt environment plumbing.

Quilt defaults to ``patches/series``; Debian keeps the series at
``debian/patches/series`` and _reset_tree removes the ``.pc`` directory
quilt would otherwise learn the location from. Without QUILT_PATCHES in the
environment every quilt call fails with "No series file found" and the
auto-refresh pass silently degrades to a no-op — observed in CI run
27377122980 where the run's initial failure was exactly the stale-patch
case preflight exists to fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ceph_autobuild_resolver import preflight
from ceph_autobuild_resolver.lxd import ExecResult


@dataclass
class _RecordingLXD:
    """Stub that scripts per-command results and records (argv, env)."""

    drift_patches: set[str] = field(default_factory=set)
    calls: list[tuple[str, dict | None]] = field(default_factory=list)
    pushes: int = 0

    timeouts: list[tuple[str, int | None]] = field(default_factory=list)

    def exec(self, container, argv, *, env=None, check=False, cwd=None, timeout=None):
        cmd = " ".join(argv)
        self.calls.append((cmd, dict(env) if env else None))
        self.timeouts.append((cmd, timeout))
        if "command -v quilt" in cmd:
            return ExecResult(0, "/usr/bin/quilt", "")
        if "quilt push --fuzz=0" in cmd:
            self.pushes += 1
            # Simulate drift on selected pushes: strict push fails.
            if f"p{self.pushes}" in self.drift_patches:
                return ExecResult(1, "", "Hunk #1 FAILED")
            return ExecResult(0, "", "")
        if "quilt push -f" in cmd or "quilt refresh" in cmd:
            return ExecResult(0, "", "")
        # patch --dry-run (phase 1), tree resets, everything else: succeed.
        return ExecResult(0, "", "")

    def put_text(self, container, path, content):  # pragma: no cover
        pass


def _quilt_calls(lxd: _RecordingLXD) -> list[tuple[str, dict | None]]:
    # Phase-2 quilt operations only. The tree-reset script's ``quilt pop``
    # needs no env: with .pc present quilt reads .pc/.quilt_patches, and
    # without it the pop is a ``|| true`` no-op.
    return [
        (cmd, env)
        for cmd, env in lxd.calls
        if "quilt push" in cmd or "quilt refresh" in cmd
    ]


def test_phase2_quilt_calls_carry_quilt_patches_env(cfg):
    lxd = _RecordingLXD()
    result = preflight.run(lxd, "ceph-build", cfg, "p1\np2\n")

    calls = _quilt_calls(lxd)
    assert calls, "phase 2 never invoked quilt"
    for cmd, env in calls:
        assert env is not None and env.get("QUILT_PATCHES") == "debian/patches", cmd
    assert "No series file found" not in result.report


def test_phase2_refresh_path_also_carries_env(cfg):
    lxd = _RecordingLXD(drift_patches={"p1"})
    result = preflight.run(lxd, "ceph-build", cfg, "p1\np2\n")

    cmds = [cmd for cmd, _ in _quilt_calls(lxd)]
    assert any("quilt push -f" in c for c in cmds)
    assert any("quilt refresh" in c for c in cmds)
    for cmd, env in _quilt_calls(lxd):
        assert env is not None and env.get("QUILT_PATCHES") == "debian/patches", cmd
    assert result.refreshed == ["p1"]


def test_unsafe_series_entries_are_skipped(cfg):
    """A series entry with path traversal must never reach the phase-1
    `patch --dry-run` redirect or _drop_patch's `rm`; the legitimate entry is
    still walked."""
    lxd = _RecordingLXD()
    preflight.run(lxd, "ceph-build", cfg, "../../etc/passwd\ngood.patch\n")
    assert not any("etc/passwd" in cmd for cmd, _ in lxd.calls)
    assert any("good.patch" in cmd for cmd, _ in lxd.calls)


def test_phase2_exec_calls_carry_a_timeout(cfg):
    """Every preflight container exec must pass a timeout: the quilt loop runs
    before the loop's wall-clock budget is armed, so an unguarded hung command
    would stall the whole run (the d649a99 timeout sweep missed preflight)."""
    lxd = _RecordingLXD()
    preflight.run(lxd, "ceph-build", cfg, "p1\np2\n")
    # No container exec in preflight may be unbounded.
    unbounded = [cmd for cmd, t in lxd.timeouts if t is None]
    assert unbounded == [], f"unbounded preflight execs: {unbounded}"
