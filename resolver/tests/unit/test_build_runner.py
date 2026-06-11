"""Tests for BuildRunner._run_stage and the public build methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

import pytest

from ceph_autobuild_resolver.build_runner import BuildRunner
from ceph_autobuild_resolver.build_steps import Stage, Step
from ceph_autobuild_resolver.lxd import ExecResult


@dataclass
class _MinimalLXD:
    """Minimal LXD stand-in for build_runner tests.

    ``step_results`` is a list of ``ExecResult`` values consumed in order for
    each ``exec`` call.  If the list runs out, returns success.
    Written log files are captured in ``logs``.
    """

    step_results: list[ExecResult] = field(default_factory=list)
    calls: list[tuple[list[str], str | None]] = field(default_factory=list)
    logs: dict[str, str] = field(default_factory=dict)

    def exec(
        self,
        container: str,
        argv: list[str],
        *,
        env: Mapping[str, str] | None = None,
        check: bool = False,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> ExecResult:
        self.calls.append((list(argv), cwd))
        if self.step_results:
            return self.step_results.pop(0)
        return ExecResult(0, f"ok: {argv[0]}\n", "")

    def exec_shell(self, container, script, *, env=None, check=False, cwd=None, timeout=None):
        return self.exec(container, ["bash", "-c", script], cwd=cwd)

    def put_text(self, container: str, remote_path: str, content: str) -> None:
        self.logs[remote_path] = content


CONTAINER = "test-container"


def _runner(cfg, lxd):
    return BuildRunner(lxd, cfg)


def test_run_stage_success_executes_all_steps(cfg):
    lxd = _MinimalLXD()
    runner = _runner(cfg, lxd)
    stage = Stage(
        name="test",
        steps=[
            Step(["echo", "one"]),
            Step(["echo", "two"]),
            Step(["echo", "three"]),
        ],
    )
    result = runner._run_stage(CONTAINER, stage)

    assert result.ok
    assert len(lxd.calls) == 3
    assert lxd.calls[0][0] == ["echo", "one"]
    assert lxd.calls[2][0] == ["echo", "three"]


def test_run_stage_stops_at_failure(cfg):
    lxd = _MinimalLXD(
        step_results=[
            ExecResult(0, "ok\n", ""),
            ExecResult(1, "", "oops"),
            ExecResult(0, "skipped\n", ""),
        ]
    )
    runner = _runner(cfg, lxd)
    stage = Stage(
        name="test",
        steps=[
            Step(["step1"]),
            Step(["step2"]),
            Step(["step3"]),
        ],
    )
    result = runner._run_stage(CONTAINER, stage)

    assert not result.ok
    assert result.returncode == 1
    assert len(lxd.calls) == 2  # step3 never ran


def test_run_stage_allow_failure_continues(cfg):
    lxd = _MinimalLXD(
        step_results=[
            ExecResult(1, "", "ignored"),
            ExecResult(0, "ran\n", ""),
        ]
    )
    runner = _runner(cfg, lxd)
    stage = Stage(
        name="test",
        steps=[
            Step(["optional"], allow_failure=True),
            Step(["required"]),
        ],
    )
    result = runner._run_stage(CONTAINER, stage)

    assert result.ok
    assert len(lxd.calls) == 2


def test_run_stage_log_written_after_each_step(cfg):
    lxd = _MinimalLXD()
    runner = _runner(cfg, lxd)
    log_path = f"{cfg.container_log_dir}/test.log"
    stage = Stage(
        name="test",
        steps=[Step(["a"]), Step(["b"])],
    )
    runner._run_stage(CONTAINER, stage)

    assert log_path in lxd.logs
    log_content = lxd.logs[log_path]
    assert "=== step: ['a']" in log_content
    assert "=== step: ['b']" in log_content


def test_run_stage_passes_cwd(cfg):
    lxd = _MinimalLXD()
    runner = _runner(cfg, lxd)
    stage = Stage(
        name="test",
        steps=[
            Step(["cmd1"], workdir="/some/path"),
            Step(["cmd2"], workdir=None),
        ],
    )
    runner._run_stage(CONTAINER, stage)

    assert lxd.calls[0][1] == "/some/path"
    assert lxd.calls[1][1] is None


def test_build_returns_outcome_on_success(cfg):
    lxd = _MinimalLXD()
    runner = _runner(cfg, lxd)
    outcome = runner.build(CONTAINER)

    assert outcome.ok
    assert outcome.stage == "build"
    assert outcome.returncode == 0
    assert outcome.log_path == f"{cfg.container_log_dir}/build.log"
    assert isinstance(outcome.log_tail, str)


def test_build_returns_outcome_on_failure(cfg):
    # Make the debuild step fail (second step after mkdir -p).
    lxd = _MinimalLXD(
        step_results=[
            ExecResult(0, "", ""),   # mkdir -p
            ExecResult(2, "build output\n", "error\n"),  # debuild
        ]
    )
    runner = _runner(cfg, lxd)
    outcome = runner.build(CONTAINER)

    assert not outcome.ok
    assert outcome.returncode == 2


def test_build_log_excerpt_focuses_on_first_error():
    """When an error/failed/fatal marker is present, the excerpt should
    surface ±50 lines around the first hit instead of just the raw tail."""
    from ceph_autobuild_resolver.build_runner import _build_log_excerpt

    lines: list[str] = []
    lines.extend(f"prelude {i}" for i in range(200))
    lines.append("CMake Error: Boost version too old")
    lines.extend(f"junk {i}" for i in range(800))

    excerpt = _build_log_excerpt("\n".join(lines))

    assert "first error context" in excerpt
    assert "CMake Error: Boost version too old" in excerpt
    assert "trailing tail" in excerpt
    # The 'prelude' and 'junk' filler far from the error must not be in
    # the excerpt — that's the whole point of the focus.
    assert "prelude 0" not in excerpt
    assert "junk 100" not in excerpt


def test_build_log_excerpt_is_case_insensitive():
    from ceph_autobuild_resolver.build_runner import _build_log_excerpt

    lines = ["context line"] * 60 + ["FAILED to link foo.o"] + ["context line"] * 60
    excerpt = _build_log_excerpt("\n".join(lines))
    assert "FAILED to link foo.o" in excerpt
    assert "first error context" in excerpt


def test_build_log_excerpt_falls_back_to_flat_tail_when_no_match():
    """If no error/failed/fatal marker is present, return the last N lines
    flat — same behaviour as before this change."""
    from ceph_autobuild_resolver.build_runner import _build_log_excerpt

    output = "\n".join(f"line {i}" for i in range(1000))
    excerpt = _build_log_excerpt(output)

    assert "first error context" not in excerpt
    # Last lines must be present.
    assert "line 999" in excerpt
    # First lines must NOT (we returned only the tail).
    assert "line 0\n" not in excerpt


def _cmake_boost_fail_log() -> str:
    """A realistic CMake configure-failure log: a long run of benign ``-- ``
    probe noise (including ``- Failed`` results) followed by the real error."""
    lines = ["-- The CXX compiler identification is GNU 14.2.0"]
    # Benign probe results that legitimately contain "Failed" — these used to
    # win the first-match scan and anchor the excerpt on configure noise.
    lines.append("-- Performing Test HAVE_REENTRANT_STRERROR_R - Failed")
    lines += [f"-- Looking for include file {h} - found"
              for h in ("sys/prctl.h", "execinfo.h", "sched.h")]
    lines += [f"-- Check size of {t} - done"
              for t in ("__u8", "__u16", "__u32", "__u64")]
    lines.append("-- Performing Test COMPILER_HAS_DEPRECATED_ATTR - Failed")
    # Push the real error well past the ±50 context window of the noise above,
    # so that anchoring on the noise would *exclude* the real error.
    lines += [f"-- filler status line {i}" for i in range(120)]
    lines += [
        "CMake Error at /usr/lib/x86_64-linux-gnu/cmake/Boost-1.90.0/BoostConfig.cmake:141 (find_package):",
        '  Could not find a package configuration file provided by "boost_system"',
        '  (requested version 1.90.0) with any of the following names:',
        "",
        "    boost_systemConfig.cmake",
        "Call Stack (most recent call first):",
        "  cmake/modules/FindBoost.cmake:2415 (find_package_handle_standard_args)",
        "  CMakeLists.txt:753 (find_package)",
        "",
        '  Could NOT find Boost (missing: system) (found suitable version "1.90.0")',
        "-- Configuring incomplete, errors occurred!",
    ]
    return "\n".join(lines)


def test_excerpt_skips_cmake_probe_noise_and_anchors_on_real_error():
    """A '-- Performing Test ... - Failed' probe must not anchor the excerpt;
    the real 'CMake Error' should, so 'missing: system' is visible inline."""
    from ceph_autobuild_resolver.build_runner import _build_log_excerpt

    excerpt = _build_log_excerpt(_cmake_boost_fail_log())
    assert "CMake Error at" in excerpt
    assert "missing: system" in excerpt
    assert "CMakeLists.txt:753" in excerpt
    # The benign probe result and far-away filler must NOT be in the window —
    # that's proof the anchor moved past the configure noise.
    assert "HAVE_REENTRANT_STRERROR_R - Failed" not in excerpt
    assert "filler status line 0" not in excerpt


def test_first_error_line_prefers_real_cmake_error_over_probe():
    from ceph_autobuild_resolver.build_runner import first_error_line

    line = first_error_line(_cmake_boost_fail_log())
    assert line.startswith("CMake Error at")
    assert "- Failed" not in line


def test_first_error_line_picks_first_compile_error_as_root_cause():
    """For compile failures the first 'error:' is the root cause; later ones
    cascade. The make '*** Error' terminator must not win over it."""
    from ceph_autobuild_resolver.build_runner import first_error_line

    log = "\n".join([
        "[ 50%] Building CXX object src/common/foo.cc.o",
        "/root/ceph/src/common/foo.cc:42:7: error: 'bar' was not declared in this scope",
        "/root/ceph/src/common/foo.cc:99:3: error: expected ';' before '}' token",
        "make[2]: *** [src/common/foo.cc.o] Error 1",
        "make[1]: *** [all] Error 2",
    ])
    assert first_error_line(log) == (
        "/root/ceph/src/common/foo.cc:42:7: error: 'bar' was not declared in this scope"
    )


def test_probe_only_log_has_no_strong_anchor():
    """A log of nothing but benign '-- ... - Failed' probes has no real error;
    the excerpt should fall back to the flat tail rather than anchor on noise."""
    from ceph_autobuild_resolver.build_runner import _build_log_excerpt

    log = "\n".join(f"-- Performing Test FEATURE_{i} - Failed" for i in range(10))
    excerpt = _build_log_excerpt(log)
    assert "first error context" not in excerpt


def test_build_log_tail_truncated(cfg):
    many_lines = "\n".join(f"line {i}" for i in range(1000))
    lxd = _MinimalLXD(
        step_results=[
            ExecResult(0, "", ""),
            ExecResult(0, many_lines, ""),
        ]
    )
    runner = _runner(cfg, lxd)
    outcome = runner.build(CONTAINER)

    assert outcome.log_tail.count("\n") <= 500


def test_build_steps_request_build_timeout(fake_lxd, cfg):
    """Every build step must carry the generous build-step timeout so a hung
    debuild cannot stall the loop forever."""
    from ceph_autobuild_resolver.build_runner import BuildRunner
    from ceph_autobuild_resolver.config import BUILD_STEP_TIMEOUT_SECONDS

    runner = BuildRunner(fake_lxd, cfg)
    runner.build("ceph-build")
    step_timeouts = [t for (cmd, t) in fake_lxd.exec_timeouts if cmd not in ("bash",)]
    assert step_timeouts, "no build steps executed"
    assert all(t == BUILD_STEP_TIMEOUT_SECONDS for t in step_timeouts)
