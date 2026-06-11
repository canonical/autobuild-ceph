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
