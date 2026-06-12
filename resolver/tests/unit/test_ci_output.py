"""CI output: workflow-command neutralization and recommendations."""

from __future__ import annotations

import re

from ceph_autobuild_resolver.output.ci_output import (
    CIFailurePayload,
    CISuccessPayload,
    _RECOMMENDATIONS,
    emit_failure,
    emit_success,
)

_HOSTILE_SUMMARY = "fixed it\n::error::spoofed annotation\n::add-mask::x"


def _success_payload() -> CISuccessPayload:
    return CISuccessPayload(
        matrix_name="resolute",
        summary=_HOSTILE_SUMMARY,
        diff="diff --git a/debian/rules b/debian/rules\n",
        transcript_path="t.jsonl",
    )


def test_success_output_neutralized_under_actions(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("CI_DIFF_FILE", raising=False)
    emit_success(_success_payload())
    out = capsys.readouterr().out

    m = re.search(r"::stop-commands::([0-9a-f]{32})", out)
    assert m, "stop-commands marker missing"
    token = m.group(1)
    stop_at = out.index(f"::stop-commands::{token}")
    end_at = out.index(f"::{token}::")
    # All model-controlled text sits inside the neutralized section.
    assert stop_at < out.index("::error::spoofed") < end_at
    assert stop_at < out.index("diff --git") < end_at
    # Token appears exactly twice: open and close.
    assert out.count(token) == 2


def test_failure_output_neutralized_under_actions(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    emit_failure(
        CIFailurePayload(
            matrix_name="resolute",
            stop_reason="max_iterations",
            failing_command="build",
            error_tail="error: x\n::error::spoofed",
            transcript_path="t.jsonl",
        )
    )
    out = capsys.readouterr().out
    m = re.search(r"::stop-commands::([0-9a-f]{32})", out)
    assert m
    token = m.group(1)
    assert out.index(f"::stop-commands::{token}") < out.index("::error::spoofed")
    assert out.index("::error::spoofed") < out.index(f"::{token}::")


def test_no_markers_outside_actions(monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("CI_DIFF_FILE", raising=False)
    emit_success(_success_payload())
    assert "::stop-commands::" not in capsys.readouterr().out


def test_diff_file_written_utf8(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CI_DIFF_FILE", str(tmp_path / "d.patch"))
    payload = _success_payload()
    payload.diff = "diff --git a/debian/changelog b/debian/changelog\n+Jørgen\n"
    emit_success(payload)
    assert "Jørgen" in (tmp_path / "d.patch").read_text(encoding="utf-8")


def test_every_loop_stop_reason_has_a_recommendation():
    # Stop reasons produced by budget.reason_for_stop(), the loop, and the
    # orchestrator/cli. A missing key silently falls back to the generic
    # recommendation, which is exactly what the review flagged.
    for reason in (
        "validation_failed",
        "max_iterations",
        "wall_time_exceeded",
        "compilation_not_reached_in_time",
        "no_progress",
        "token_budget_exceeded",
        "declared_unresolvable",
        "context_overflow",
        "provider_error",
        "resolver_crash",
    ):
        assert reason in _RECOMMENDATIONS, reason


def test_write_diff_artifact_writes_when_env_set(monkeypatch, tmp_path):
    from ceph_autobuild_resolver.output.ci_output import write_diff_artifact

    target = tmp_path / "out.patch"
    monkeypatch.setenv("CI_DIFF_FILE", str(target))
    assert write_diff_artifact("diff --git a/x b/x\n+Jørgen\n") is True
    assert target.read_text(encoding="utf-8").startswith("diff --git")
    assert "Jørgen" in target.read_text(encoding="utf-8")


def test_write_diff_artifact_noop_without_env(monkeypatch):
    from ceph_autobuild_resolver.output.ci_output import write_diff_artifact

    monkeypatch.delenv("CI_DIFF_FILE", raising=False)
    assert write_diff_artifact("anything") is False


def test_validation_error_has_recommendation():
    assert "validation_error" in _RECOMMENDATIONS
