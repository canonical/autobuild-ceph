from __future__ import annotations

from pathlib import Path

import pytest

from ceph_autobuild_resolver.build_runner import BuildRunner
from ceph_autobuild_resolver.providers.base import ToolCall
from ceph_autobuild_resolver.tools.dispatch import Dispatcher
from ceph_autobuild_resolver.tools.execution import ExecutionHandlers
from ceph_autobuild_resolver.tools.filesystem import FilesystemHandlers
from ceph_autobuild_resolver.tools.search import SearchHandlers


@pytest.fixture
def dispatcher(fake_lxd, cfg) -> Dispatcher:
    runner = BuildRunner(fake_lxd, cfg)
    fs = FilesystemHandlers(
        lxd=fake_lxd, cfg=cfg, container="ceph-build", runner=runner
    )
    sr = SearchHandlers(lxd=fake_lxd, cfg=cfg, container="ceph-build")
    ex = ExecutionHandlers(runner=runner, container="ceph-build")
    return Dispatcher(fs, sr, ex, build_log_path="/root/build-logs/build.log")


def test_unknown_tool_returns_error_payload(dispatcher):
    outcome = dispatcher.dispatch(
        [ToolCall(id="c1", name="bogus", args={})]
    )
    assert outcome.results[0].payload == {"error": "unknown tool: bogus"}
    assert outcome.declared_resolved is False


def test_malformed_arguments_returned_as_data(dispatcher):
    outcome = dispatcher.dispatch(
        [
            ToolCall(
                id="c1",
                name="grep",
                args={"__malformed_arguments__": "{not json"},
            )
        ]
    )
    payload = outcome.results[0].payload
    assert payload["error"] == "your tool arguments were not valid JSON; re-issue with valid JSON"


def test_scope_violation_returned_as_structured_error(dispatcher):
    outcome = dispatcher.dispatch(
        [
            ToolCall(
                id="c1",
                name="write_file",
                args={"path": "src/foo.cc", "content": "x"},
            )
        ]
    )
    p = outcome.results[0].payload
    assert p["error"] == "edit_scope_violation"
    assert "outside debian/" in p["message"]


def test_bad_arguments_caught(dispatcher):
    outcome = dispatcher.dispatch(
        [ToolCall(id="c1", name="grep", args={"not_a_real_arg": True})]
    )
    p = outcome.results[0].payload
    assert p["error"] == "bad_arguments"


def test_run_build_refused_when_no_changes_since_last_failure(dispatcher, fake_lxd, cfg):
    """The hard guard should refuse a second run_build when the previous
    one failed and no file mutator has succeeded in between."""
    from ceph_autobuild_resolver.build_runner import BuildOutcome

    # Pretend a previous build failed.
    dispatcher._execution.last_build = BuildOutcome(
        ok=False,
        stage="build",
        returncode=2,
        log_path="/root/build-logs/build.log",
        log_tail="some error",
    )
    dispatcher._execution.files_changed_since_last_build = False

    outcome = dispatcher.dispatch([ToolCall(id="c1", name="run_build", args={})])
    payload = outcome.results[0].payload

    assert payload["skipped"] is True
    assert payload["error"] == "no_changes_since_last_build"
    # Real build was NOT invoked — fake_lxd should have logged nothing.
    assert all("debuild" not in " ".join(argv) for _, argv in fake_lxd.exec_log)


def test_run_build_proceeds_after_mutation(dispatcher, fake_lxd):
    """A successful file mutator in the SAME dispatch turn should clear the
    debt and let run_build proceed."""
    from ceph_autobuild_resolver.build_runner import BuildOutcome

    dispatcher._execution.last_build = BuildOutcome(
        ok=False,
        stage="build",
        returncode=2,
        log_path="/root/build-logs/build.log",
        log_tail="some error",
    )
    dispatcher._execution.files_changed_since_last_build = False

    outcome = dispatcher.dispatch(
        [
            ToolCall(
                id="c1",
                name="write_file",
                args={"path": "debian/foo", "content": "x"},
            ),
            ToolCall(id="c2", name="run_build", args={}),
        ]
    )
    # write_file came first → flag set → run_build should NOT be skipped.
    run_payload = outcome.results[1].payload
    assert "skipped" not in run_payload or run_payload.get("skipped") is not True


def test_grep_log_routed_with_injected_log_path(dispatcher, fake_lxd):
    """The dispatcher must inject the build log path so the model never
    has to know it (mirrors how read_log is wired)."""
    fake_lxd.put_text(
        "ceph-build", "/root/build-logs/build.log", "ok\nCMake Error: bad\nmore\n"
    )
    outcome = dispatcher.dispatch(
        [ToolCall(id="c1", name="grep_log", args={"pattern": "CMake Error"})]
    )
    payload = outcome.results[0].payload
    assert payload.get("any_matches") is True
    assert "CMake Error: bad" in payload.get("output", "")


def test_oversized_handler_result_is_clamped(dispatcher):
    """Any handler payload over the byte ceiling is replaced with a stub."""
    from ceph_autobuild_resolver.tools.dispatch import MAX_PAYLOAD_BYTES

    dispatcher._handlers["grep"] = lambda **kw: {
        "matches": "x" * (MAX_PAYLOAD_BYTES + 1024)
    }
    outcome = dispatcher.dispatch(
        [ToolCall(id="c1", name="grep", args={"pattern": "p"})]
    )
    p = outcome.results[0].payload
    assert p["error"] == "result_truncated"
    assert p["total_bytes"] > MAX_PAYLOAD_BYTES


def test_oversized_non_handler_branch_is_clamped(dispatcher):
    """The clamp covers branches that bypass the handler path -- here the
    malformed-args echo, which reflects the raw model blob back."""
    from ceph_autobuild_resolver.tools.dispatch import MAX_PAYLOAD_BYTES

    huge = "{" + "x" * (MAX_PAYLOAD_BYTES + 1024)
    outcome = dispatcher.dispatch(
        [ToolCall(id="c1", name="grep", args={"__malformed_arguments__": huge})]
    )
    p = outcome.results[0].payload
    assert p["error"] == "result_truncated"


def test_oversized_mutator_result_preserves_files_changed(dispatcher):
    """Clamping a large *successful* mutator payload must not undo the
    files_changed bookkeeping, which runs before the post-loop clamp."""
    from ceph_autobuild_resolver.tools.dispatch import MAX_PAYLOAD_BYTES

    dispatcher._handlers["write_file"] = lambda **kw: {
        "ok": True,
        "path": kw.get("path"),
        "blob": "x" * (MAX_PAYLOAD_BYTES + 1024),
    }
    dispatcher._execution.files_changed_since_last_build = False

    outcome = dispatcher.dispatch(
        [
            ToolCall(
                id="c1",
                name="write_file",
                args={"path": "debian/foo", "content": "y"},
            )
        ]
    )
    # Bookkeeping observed the real ok=True payload before truncation.
    assert dispatcher.files_changed is True
    p = outcome.results[0].payload
    assert p["error"] == "result_truncated"
    # Small scalar keys are preserved for the model's benefit.
    assert p["ok"] is True
    assert p["path"] == "debian/foo"


def test_declare_resolved_signals_termination(dispatcher):
    outcome = dispatcher.dispatch(
        [
            ToolCall(
                id="c1",
                name="declare_resolved",
                args={"summary": "fixed it"},
            )
        ]
    )
    assert outcome.declared_resolved is True
    assert outcome.resolution_summary == "fixed it"
