"""Smoke tests: untrusted text must never crash Rich rendering.

Build logs and model output routinely contain bracketed paths like
``[/usr/lib/ceph]`` which Rich's markup parser treats as a closing tag and
rejects with MarkupError. Every display helper that renders untrusted text
must escape it first. The empirically verified crash case from review:
``model_text('checking file [/usr/lib/ceph] for symbols')``.
"""

from __future__ import annotations

from ceph_autobuild_resolver import display

_HOSTILE = "checking file [/usr/lib/ceph] for symbols [bold]x[/nope] [unclosed"


def test_model_text_survives_markup_like_input():
    display.model_text(_HOSTILE)


def test_model_reasoning_survives_markup_like_input():
    display.model_reasoning(_HOSTILE)


def test_build_result_survives_markup_like_log_tail():
    display.build_result(False, "build", _HOSTILE)
    display.build_result(True, "[stage]", _HOSTILE)


def test_step_done_survives_markup_like_output():
    display.step_done(False, returncode=2, output=_HOSTILE)


def test_tool_dispatch_survives_markup_like_args():
    display.tool_dispatch("read_files", {"paths": ["[/usr/lib]"], "x": _HOSTILE})


def test_preflight_summary_survives_markup_like_report():
    display.preflight_summary(_HOSTILE, dropped=["[a].patch"], refreshed=[])


def test_file_written_and_diff_survive_markup_like_paths():
    display.file_written("debian/[new]", None, "content\n")
    display.file_edited("debian/[edit]", "a\n", "b\n")
    display.file_edited("debian/[same]", "a\n", "a\n")


def test_run_summary_survives_markup_like_input():
    diff = (
        "diff --git a/debian/patches/[weird].patch b/debian/patches/[weird].patch\n"
        "new file mode 100644\n"
    )
    display.run_summary(
        success=False,
        stop_reason="max_iterations",
        iterations=3,
        total_tokens=1234,
        elapsed_seconds=12.0,
        initial_error=_HOSTILE + "\nline2\n",
        resolution_summary=_HOSTILE,
        diff=diff,
        last_build_error=_HOSTILE,
    )


def test_startup_info_survives_markup_like_ccache_dir():
    display.startup_info("m", "p", "/tmp/[ccache]")


def test_startup_info_survives_markup_like_model_name():
    # MODEL_NAME can contain brackets (e.g. gemini-x[preview]); must not crash.
    display.startup_info("gemini-3.1[preview-05-20]", "gemini", None)
    display.startup_info(_HOSTILE, "gemini", None)


def test_model_text_neutralizes_workflow_commands_under_actions(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    display.model_text("here is a fix\n::add-mask::secret\n::error::spoofed")
    err = capsys.readouterr().err
    import re as _re
    m = _re.search(r"::stop-commands::([0-9a-f]{32})", err)
    assert m, "stop-commands token missing on stderr"
    token = m.group(1)
    # The model's ::add-mask:: sits inside the neutralized window.
    assert err.index(f"::stop-commands::{token}") < err.index("add-mask")
    assert err.index("add-mask") < err.index(f"::{token}::")


def test_no_neutralization_markers_outside_actions(monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    display.model_text("plain output")
    assert "::stop-commands::" not in capsys.readouterr().err
