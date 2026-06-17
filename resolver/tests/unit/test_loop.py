"""End-to-end loop tests with a stubbed provider.

The provider stub plays a scripted sequence of replies so we can verify the
loop drives history, dispatch, and budget correctly without the real model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ceph_autobuild_resolver.budget import Budget
from ceph_autobuild_resolver.build_runner import BuildRunner
from ceph_autobuild_resolver.loop import (
    _COMPRESS_AFTER_TURNS,
    _KEEP_RECENT_TURNS,
    _compress_history,
    _summarize_turns,
    run_loop,
)
from ceph_autobuild_resolver.providers.base import (
    Message,
    ToolCall,
    ToolResult,
    ToolSchema,
    Usage,
)
from ceph_autobuild_resolver.tools.dispatch import Dispatcher
from ceph_autobuild_resolver.tools.execution import ExecutionHandlers
from ceph_autobuild_resolver.tools.filesystem import FilesystemHandlers
from ceph_autobuild_resolver.tools.search import SearchHandlers
from ceph_autobuild_resolver.transcript import Transcript


class ScriptedProvider:
    def __init__(self, replies: list[Message], usages: list[Usage] | None = None):
        self._replies = list(replies)
        self._usages = list(usages or [Usage(10, 5)] * len(replies))
        self.declared: list[ToolSchema] = []

    def declare_tools(self, tools):
        self.declared = list(tools)

    def chat(self, history):
        if not self._replies:
            raise AssertionError("ScriptedProvider exhausted")
        return self._replies.pop(0), self._usages.pop(0)


def _build_dispatcher(fake_lxd, cfg) -> Dispatcher:
    runner = BuildRunner(fake_lxd, cfg)
    fs = FilesystemHandlers(
        lxd=fake_lxd, cfg=cfg, container="ceph-build", runner=runner
    )
    sr = SearchHandlers(lxd=fake_lxd, cfg=cfg, container="ceph-build")
    ex = ExecutionHandlers(runner=runner, container="ceph-build")
    return Dispatcher(fs, sr, ex, build_log_path="/root/build-logs/build.log")


def _seed_clean_build(dispatcher: Dispatcher) -> None:
    """Mark the last build as a clean success so declare_resolved is accepted
    (the dispatch guard now rejects it when no build has succeeded)."""
    from ceph_autobuild_resolver.build_runner import BuildOutcome

    dispatcher._execution.last_build = BuildOutcome(
        ok=True, stage="build", returncode=0,
        log_path="/root/build-logs/build.log", log_tail="ok",
    )
    dispatcher._execution.files_changed_since_last_build = False


def test_loop_terminates_on_declare_resolved(fake_lxd, cfg, tmp_path):
    provider = ScriptedProvider(
        [
            Message(
                role="model",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="declare_resolved",
                        args={"summary": "all good"},
                    )
                ],
            )
        ]
    )
    dispatcher = _build_dispatcher(fake_lxd, cfg)
    _seed_clean_build(dispatcher)
    transcript = Transcript(tmp_path / "t.jsonl")
    budget = Budget.from_config(cfg)

    result = run_loop(
        history=[Message(role="user", text="go")],
        provider=provider,
        dispatcher=dispatcher,
        budget=budget,
        transcript=transcript,
    )

    assert result.declared_resolved is True
    assert result.resolution_summary == "all good"


def test_loop_nudges_on_textonly_reply(fake_lxd, cfg, tmp_path):
    provider = ScriptedProvider(
        [
            Message(role="model", text="thinking..."),
            Message(
                role="model",
                tool_calls=[
                    ToolCall(id="c1", name="declare_resolved", args={"summary": "ok"})
                ],
            ),
        ]
    )
    dispatcher = _build_dispatcher(fake_lxd, cfg)
    _seed_clean_build(dispatcher)
    budget = Budget.from_config(cfg)
    transcript = Transcript(tmp_path / "t.jsonl")

    result = run_loop(
        history=[Message(role="user", text="go")],
        provider=provider,
        dispatcher=dispatcher,
        budget=budget,
        transcript=transcript,
    )
    assert result.declared_resolved is True
    # The second-to-last message should be the nudge we appended.
    nudges = [
        m for m in result.history if m.role == "user" and m.text and "tools" in m.text
    ]
    assert len(nudges) >= 1


def test_loop_stops_on_iteration_cap(fake_lxd, cfg, tmp_path):
    # All-text replies — model never produces a tool call. Loop should
    # eventually exhaust max_iterations.
    replies = [Message(role="model", text="...") for _ in range(cfg.max_iterations + 2)]
    provider = ScriptedProvider(replies)
    dispatcher = _build_dispatcher(fake_lxd, cfg)
    transcript = Transcript(tmp_path / "t.jsonl")
    budget = Budget.from_config(cfg)

    result = run_loop(
        history=[Message(role="user", text="go")],
        provider=provider,
        dispatcher=dispatcher,
        budget=budget,
        transcript=transcript,
    )
    assert result.declared_resolved is False
    assert result.stop_reason == "max_iterations"


def test_loop_terminates_quickly_on_repeated_refused_builds(fake_lxd, cfg, tmp_path):
    """Refused builds (hard guard) should count as no-progress evidence and
    drive the unchanged_streak just like a failed build with no changes —
    otherwise a zombie model can hammer run_build until max_iterations.
    """
    from ceph_autobuild_resolver.build_runner import BuildOutcome

    # Pretend a real failed build already happened so the guard will fire.
    dispatcher = _build_dispatcher(fake_lxd, cfg)
    dispatcher._execution.last_build = BuildOutcome(
        ok=False,
        stage="build",
        returncode=2,
        log_path="/root/build-logs/build.log",
        log_tail="some error",
    )
    dispatcher._execution.files_changed_since_last_build = False

    # Script: enough run_build calls to exceed the iteration cap, but the
    # streak should bite first.
    replies = [
        Message(
            role="model",
            tool_calls=[ToolCall(id=f"c{i}", name="run_build", args={})],
        )
        for i in range(cfg.max_iterations + 5)
    ]
    provider = ScriptedProvider(replies)
    transcript = Transcript(tmp_path / "t.jsonl")
    budget = Budget.from_config(cfg)

    result = run_loop(
        history=[Message(role="user", text="go")],
        provider=provider,
        dispatcher=dispatcher,
        budget=budget,
        transcript=transcript,
    )
    assert result.declared_resolved is False
    assert result.stop_reason == "no_progress"
    # And we exited well before the iteration cap.
    assert budget.iterations_used <= cfg.max_unchanged_iterations + 1


def test_loop_exits_clean_on_context_overflow(fake_lxd, cfg, tmp_path):
    """A ContextOverflowError from the provider must stop the loop cleanly
    with stop_reason='context_overflow' -- no exception escapes, and there is
    NO retry (compression can't evict the offending last message, so a retry
    would just overflow again)."""
    from ceph_autobuild_resolver.providers.base import ContextOverflowError

    class OverflowProvider:
        def __init__(self):
            self.calls = 0

        def declare_tools(self, tools):
            pass

        def chat(self, history):
            self.calls += 1
            raise ContextOverflowError(
                "input token count exceeds the maximum number of tokens allowed 1048576"
            )

    provider = OverflowProvider()
    dispatcher = _build_dispatcher(fake_lxd, cfg)
    budget = Budget.from_config(cfg)
    transcript = Transcript(tmp_path / "t.jsonl")

    result = run_loop(
        history=[Message(role="user", text="go")],
        provider=provider,
        dispatcher=dispatcher,
        budget=budget,
        transcript=transcript,
    )

    assert result.declared_resolved is False
    assert result.stop_reason == "context_overflow"
    assert provider.calls == 1  # no retry


def test_loop_clamps_oversized_dispatcher_crash_payload(
    fake_lxd, cfg, tmp_path, monkeypatch
):
    """If the dispatcher itself raises with a huge message, the synthetic
    crash result must still go through the byte ceiling -- this is the one
    result-append path outside dispatch()."""
    from ceph_autobuild_resolver.tools.dispatch import MAX_PAYLOAD_BYTES

    provider = ScriptedProvider(
        [
            Message(
                role="model",
                tool_calls=[ToolCall(id="c1", name="grep", args={"pattern": "x"})],
            ),
            Message(
                role="model",
                tool_calls=[
                    ToolCall(id="c2", name="declare_resolved", args={"summary": "ok"})
                ],
            ),
        ]
    )
    dispatcher = _build_dispatcher(fake_lxd, cfg)

    calls = {"n": 0}
    real_dispatch = dispatcher.dispatch

    def boom_once(tool_calls):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("x" * (MAX_PAYLOAD_BYTES + 1024))
        return real_dispatch(tool_calls)

    monkeypatch.setattr(dispatcher, "dispatch", boom_once)
    budget = Budget.from_config(cfg)
    transcript = Transcript(tmp_path / "t.jsonl")

    result = run_loop(
        history=[Message(role="user", text="go")],
        provider=provider,
        dispatcher=dispatcher,
        budget=budget,
        transcript=transcript,
    )

    payloads = [
        r.payload
        for m in result.history
        if m.role == "tool"
        for r in m.tool_results
    ]
    # The crash payload was oversized and must have been clamped to the stub.
    assert any(p.get("error") == "result_truncated" for p in payloads)
    assert not any(
        p.get("error") == "dispatcher_crash" and p.get("total_bytes") is None
        for p in payloads
    )


def test_loop_records_transcript(fake_lxd, cfg, tmp_path):
    provider = ScriptedProvider(
        [
            Message(
                role="model",
                tool_calls=[
                    ToolCall(id="c1", name="declare_resolved", args={"summary": "ok"})
                ],
            )
        ]
    )
    dispatcher = _build_dispatcher(fake_lxd, cfg)
    transcript_path = tmp_path / "t.jsonl"
    transcript = Transcript(transcript_path)
    budget = Budget.from_config(cfg)

    run_loop(
        history=[Message(role="user", text="go")],
        provider=provider,
        dispatcher=dispatcher,
        budget=budget,
        transcript=transcript,
    )
    lines = transcript_path.read_text().strip().splitlines()
    kinds = [line for line in lines if line]
    assert len(kinds) >= 2  # model_turn + tool_results at minimum


# ---------------------------------------------------------------------------
# History compression
# ---------------------------------------------------------------------------


def _make_history(n_pairs: int) -> list[Message]:
    """Build a history with system + initial user + n_pairs of model/tool msgs."""
    history = [
        Message(role="system", text="sys"),
        Message(role="user", text="initial"),
    ]
    for i in range(n_pairs):
        history.append(
            Message(
                role="model",
                tool_calls=[ToolCall(id=f"c{i}", name="run_build", args={})],
            )
        )
        history.append(
            Message(
                role="tool",
                tool_results=[
                    ToolResult(
                        call_id=f"c{i}",
                        name="run_build",
                        payload={"ok": False, "stage": "build", "returncode": 1, "log_tail": "error: broken"},
                    )
                ],
            )
        )
    return history


def test_compress_history_noop_when_short():
    history = _make_history(5)
    original_len = len(history)
    assert original_len <= _COMPRESS_AFTER_TURNS
    _compress_history(history)
    assert len(history) == original_len


def test_compress_history_replaces_middle_turns():
    # Build a history long enough to trigger compression.
    n_pairs = (_COMPRESS_AFTER_TURNS // 2) + 2
    history = _make_history(n_pairs)
    assert len(history) > _COMPRESS_AFTER_TURNS

    original_id = id(history)
    _compress_history(history)

    # Still the same list object (mutated in place).
    assert id(history) == original_id
    # system + initial are untouched.
    assert history[0].role == "system"
    assert history[0].text == "sys"
    assert history[1].role == "user"
    assert history[1].text == "initial"
    # history[2] is now the synthetic summary.
    assert history[2].role == "user"
    assert "compressed" in (history[2].text or "").lower()
    # Total length is bounded.
    assert len(history) <= 2 + 1 + _KEEP_RECENT_TURNS


def test_compress_history_never_splits_model_tool_pair():
    """The cut must land on a non-tool message so we never orphan tool results."""
    history = _make_history((_COMPRESS_AFTER_TURNS // 2) + 2)
    _compress_history(history)
    # The first message after the summary must not be a bare tool message.
    assert history[3].role != "tool"


def test_compress_history_preserves_recent_turns():
    n_pairs = (_COMPRESS_AFTER_TURNS // 2) + 2
    history = _make_history(n_pairs)
    # Stamp the last few messages so we can verify they survive.
    history[-1].text = "SENTINEL"
    _compress_history(history)
    last_texts = [m.text for m in history]
    assert "SENTINEL" in last_texts


def test_compress_history_carries_prior_summary_forward():
    """A second compression must not erase the first compression's summary:
    its bullet lines are carried into the new summary."""
    from ceph_autobuild_resolver.loop import _compress_history

    history = _make_history((_COMPRESS_AFTER_TURNS // 2) + 2)
    # Make the earliest turn distinctive so we can trace it through both
    # compressions.
    history[2].tool_calls = [
        ToolCall(id="e0", name="edit_file", args={"path": "debian/earliest-edit"})
    ]
    _compress_history(history)
    first_summary = history[2].text
    assert "debian/earliest-edit" in first_summary

    # Grow past the threshold again and re-compress.
    history.extend(_make_history((_COMPRESS_AFTER_TURNS // 2) + 2)[2:])
    _compress_history(history)
    second_summary = history[2].text
    assert second_summary != first_summary
    assert "debian/earliest-edit" in second_summary


def test_compress_history_ignores_ordinary_user_message_at_cut():
    from ceph_autobuild_resolver.loop import _compress_history

    history = _make_history((_COMPRESS_AFTER_TURNS // 2) + 2)
    history.insert(2, Message(role="user", text="please fix the build"))
    _compress_history(history)
    # The ordinary user message is summarized away, not treated as a
    # prior summary whose lines get carried forward.
    assert "please fix the build" not in (history[2].text or "")


def test_summarize_turns_uses_real_schema_arg_keys():
    """apply_patch's parameter is the diff text and replace_in_upstream's
    file key is 'file' -- the old summary code used non-existent keys and
    rendered '?' for both."""
    diff = "--- a/debian/rules\n+++ b/debian/rules\n@@ -1 +1 @@\n-a\n+b\n"
    turns = [
        Message(
            role="model",
            tool_calls=[
                ToolCall(id="c1", name="apply_patch", args={"diff": diff}),
                ToolCall(
                    id="c2",
                    name="replace_in_upstream",
                    args={"file": "src/common/Foo.cc", "patch_name": "x"},
                ),
            ],
        )
    ]
    summary = _summarize_turns(turns)
    assert "debian/rules" in summary
    assert "src/common/Foo.cc" in summary
    assert "?" not in summary


def test_summarize_turns_includes_build_outcome():
    turns = [
        Message(
            role="model",
            tool_calls=[ToolCall(id="c1", name="run_build", args={})],
        ),
        Message(
            role="tool",
            tool_results=[
                ToolResult(
                    call_id="c1",
                    name="run_build",
                    payload={
                        "ok": False,
                        "stage": "build",
                        "returncode": 2,
                        "log_tail": "fatal error: foo.h not found",
                    },
                )
            ],
        ),
    ]
    summary = _summarize_turns(turns)
    assert "FAILED" in summary
    assert "foo.h not found" in summary


def test_summarize_turns_includes_file_edits():
    turns = [
        Message(
            role="model",
            tool_calls=[
                ToolCall(id="c1", name="edit_file", args={"path": "debian/rules"})
            ],
        ),
    ]
    summary = _summarize_turns(turns)
    assert "debian/rules" in summary


def test_loop_compresses_history_when_long(fake_lxd, cfg, tmp_path):
    """The loop should compress history before it grows unboundedly."""
    # Override max_iterations high enough to trigger compression.
    from dataclasses import replace as dc_replace

    long_cfg = dc_replace(cfg, max_iterations=_COMPRESS_AFTER_TURNS + 5)
    budget = Budget.from_config(long_cfg)

    # All text-only replies so the loop keeps running until the iteration cap.
    replies = [
        Message(role="model", text="thinking...")
        for _ in range(long_cfg.max_iterations + 2)
    ]
    provider = ScriptedProvider(replies)
    dispatcher = _build_dispatcher(fake_lxd, long_cfg)
    transcript = Transcript(tmp_path / "t.jsonl")

    result = run_loop(
        history=[
            Message(role="system", text="sys"),
            Message(role="user", text="initial"),
        ],
        provider=provider,
        dispatcher=dispatcher,
        budget=budget,
        transcript=transcript,
    )

    # Despite many iterations, history never grew past a manageable size.
    # Compression fires at the top of an iteration once len exceeds
    # _COMPRESS_AFTER_TURNS, and an iteration appends at most two messages,
    # so the history can only ever overshoot the threshold by one
    # iteration's growth. (The old bound of 2+1+keep+iters*2 was vacuously
    # true even with compression completely broken.)
    max_expected = _COMPRESS_AFTER_TURNS + 4
    assert len(result.history) <= max_expected
    # At least one compression summary should be present.
    summaries = [
        m for m in result.history
        if m.role == "user" and "compressed" in (m.text or "").lower()
    ]
    assert len(summaries) >= 1


def test_loop_stops_cleanly_on_provider_error(fake_lxd, cfg, tmp_path):
    """A non-overflow provider exception (retries already exhausted in the
    adapter) must stop the loop with stop_reason='provider_error' so the
    orchestrator records an outcome instead of crashing."""

    class FlakyProvider:
        def declare_tools(self, tools):
            pass

        def chat(self, history):
            raise RuntimeError("429 RESOURCE_EXHAUSTED after 5 attempts")

    dispatcher = _build_dispatcher(fake_lxd, cfg)
    budget = Budget.from_config(cfg)
    transcript = Transcript(tmp_path / "t.jsonl")

    result = run_loop(
        history=[Message(role="user", text="go")],
        provider=FlakyProvider(),
        dispatcher=dispatcher,
        budget=budget,
        transcript=transcript,
    )

    assert result.declared_resolved is False
    assert result.stop_reason == "provider_error"
    assert "429" in (result.resolution_summary or "")


def test_loop_propagates_keyboard_interrupt(fake_lxd, cfg, tmp_path, monkeypatch):
    """Ctrl-C during dispatch (most of the wall time) must abort the run,
    not be converted into a synthetic dispatcher_crash tool result."""
    replies = [
        Message(
            role="model",
            tool_calls=[ToolCall(id="c1", name="run_build", args={})],
        )
    ]
    provider = ScriptedProvider(replies)
    dispatcher = _build_dispatcher(fake_lxd, cfg)

    def interrupt(calls):
        raise KeyboardInterrupt

    monkeypatch.setattr(dispatcher, "dispatch", interrupt)
    budget = Budget.from_config(cfg)
    transcript = Transcript(tmp_path / "t.jsonl")

    with pytest.raises(KeyboardInterrupt):
        run_loop(
            history=[Message(role="user", text="go")],
            provider=provider,
            dispatcher=dispatcher,
            budget=budget,
            transcript=transcript,
        )
