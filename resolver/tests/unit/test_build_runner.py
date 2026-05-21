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
    ) -> ExecResult:
        self.calls.append((list(argv), cwd))
        if self.step_results:
            return self.step_results.pop(0)
        return ExecResult(0, f"ok: {argv[0]}\n", "")

    def exec_shell(self, container, script, *, env=None, check=False, cwd=None):
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
