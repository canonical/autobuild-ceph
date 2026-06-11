"""Terminal display helpers for the resolver UI.

All output goes to stderr so stdout stays clean for any machine-readable
output.  Rich auto-detects whether stderr is a TTY; in non-interactive
contexts (CI, unit tests with captured stderr) panels still render but
without ANSI colour — which is fine.
"""

from __future__ import annotations

import difflib
import re

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax

_c = Console(stderr=True, highlight=False)


# ---------------------------------------------------------------------------
# Prep / build-runner output
# ---------------------------------------------------------------------------


def startup_info(model: str, provider: str, ccache_host_dir: str | None) -> None:
    ccache = (
        f"[bold green]enabled[/bold green] ({escape(ccache_host_dir)})"
        if ccache_host_dir
        else "[dim]disabled[/dim]"
    )
    _c.print(
        f"  [bold]provider[/bold] {provider} · [bold]model[/bold] {model} · "
        f"[bold]ccache[/bold] {ccache}"
    )


def stage_header(name: str) -> None:
    _c.rule(f"[bold white]{name}[/bold white]")


def step_start(argv: list[str]) -> None:
    label = " ".join(str(a) for a in argv[:3])
    if len(argv) > 3:
        label += " …"
    _c.print(f"  [dim cyan]→ {escape(label)}[/dim cyan]")


def step_done(ok: bool, returncode: int = 0, output: str = "") -> None:
    if ok:
        _c.print("    [bold green]✓[/bold green]")
    else:
        _c.print(f"    [bold red]✗  rc={returncode}[/bold red]")
        tail = output.strip()[-2000:] if output.strip() else ""
        if tail:
            _c.print(escape(tail), style="dim red")


# ---------------------------------------------------------------------------
# Resolve-loop output
# ---------------------------------------------------------------------------


def model_text(text: str) -> None:
    if not text.strip():
        return
    # Model text and build logs are untrusted: bracketed paths like
    # [/usr/lib/ceph] are valid markup-closing tags and crash Rich.
    _c.print(
        Panel(
            escape(text.strip()),
            title="[bold cyan]model[/bold cyan]",
            border_style="cyan",
        )
    )


def model_reasoning(text: str) -> None:
    if not text.strip():
        return
    _c.print(
        Panel(
            escape(text.strip()),
            title="[bold magenta]thinking[/bold magenta]",
            border_style="magenta",
        )
    )


def token_usage(input_tokens: int, output_tokens: int, budget: int) -> None:
    total = input_tokens + output_tokens
    budget_str = f" / {budget:,} budget" if budget > 0 else ""
    _c.print(
        f"  [dim]tokens: {input_tokens:,} in · {output_tokens:,} out · "
        f"{total:,} total{budget_str}[/dim]"
    )


def tool_dispatch(name: str, args: dict) -> None:
    """One-liner shown before every tool call."""
    parts = ", ".join(
        f"[dim]{escape(str(k))}[/dim]=[cyan]{escape(repr(v)[:80])}[/cyan]"
        for k, v in list(args.items())[:4]
    )
    suffix = " …" if len(args) > 4 else ""
    _c.print(f"  [bold blue]⚙ {escape(name)}[/bold blue]({parts}{suffix})")


def file_written(path: str, old: str | None, new_content: str) -> None:
    if old is None:
        _c.print(
            Panel(
                Syntax(
                    new_content,
                    _guess_lang(path),
                    line_numbers=True,
                    word_wrap=True,
                ),
                title=f"[bold green]new file  {escape(path)}[/bold green]",
                border_style="green",
            )
        )
    else:
        _show_diff(path, old, new_content)


def file_edited(path: str, old_full: str, new_full: str) -> None:
    _show_diff(path, old_full, new_full)


def preflight_summary(report: str, dropped: list[str], refreshed: list[str]) -> None:
    """Render the preflight outcome as a panel.

    The report is the per-patch table; dropped/refreshed counts go in the
    title so the user can see at a glance whether the preflight made any
    deterministic changes.
    """
    if not report.strip():
        return
    badge_parts: list[str] = []
    if dropped:
        badge_parts.append(f"[bold red]{len(dropped)} dropped[/bold red]")
    if refreshed:
        badge_parts.append(f"[bold yellow]{len(refreshed)} refreshed[/bold yellow]")
    badge = f" — {' · '.join(badge_parts)}" if badge_parts else ""
    _c.print(
        Panel(
            escape(report),
            title=f"[bold magenta]preflight[/bold magenta]{badge}",
            border_style="magenta",
        )
    )


def build_result(ok: bool, stage: str, log_tail: str) -> None:
    color = "green" if ok else "red"
    status = "PASSED" if ok else "FAILED"
    tail = log_tail[-3000:] if log_tail else "(no output)"
    _c.print(
        Panel(
            escape(tail),
            title=f"[bold {color}]build {status} @ {escape(stage)}[/bold {color}]",
            border_style=color,
        )
    )


# ---------------------------------------------------------------------------
# End-of-run summary
# ---------------------------------------------------------------------------


def run_summary(
    *,
    success: bool,
    stop_reason: str,
    iterations: int,
    total_tokens: int,
    elapsed_seconds: float,
    initial_error: str,
    resolution_summary: str | None,
    diff: str,
    last_build_error: str = "",
) -> None:
    """Print a structured end-of-run summary to stderr."""
    elapsed = _fmt_elapsed(elapsed_seconds)
    tok = f"{total_tokens / 1_000_000:.1f}M" if total_tokens >= 1_000_000 else f"{total_tokens:,}"
    stats = f"{iterations} iter · {tok} tokens · {elapsed}"

    if success:
        title = f"[bold green]RESOLVED[/bold green]  ({stats})"
        color = "green"
    else:
        title = f"[bold red]FAILED — {stop_reason}[/bold red]  ({stats})"
        color = "red"

    lines: list[str] = []

    # Initial failure
    if initial_error:
        lines.append("[bold]Initial error:[/bold]")
        for ln in initial_error.splitlines()[:3]:
            lines.append(f"  {escape(ln.strip())}")
        lines.append("")

    # Patch-level changes parsed from the diff
    patch_changes = _parse_patch_changes(diff)
    if patch_changes:
        lines.append("[bold]Patch changes:[/bold]")
        for status, name in patch_changes:
            icon, style = {
                "added":    ("+", "green"),
                "dropped":  ("-", "red"),
                "modified": ("~", "yellow"),
            }[status]
            lines.append(f"  [{style}]{icon} {escape(name)}[/{style}]")
        lines.append("")
    elif success:
        lines.append("[dim]No patch changes.[/dim]\n")

    # Model's explanation
    if resolution_summary:
        lines.append("[bold]Model explanation:[/bold]")
        for ln in resolution_summary.strip().splitlines()[:20]:
            lines.append(f"  {escape(ln)}")
        lines.append("")

    # On failure: last known error
    if not success and last_build_error:
        lines.append("[bold]Last build error:[/bold]")
        lines.append(f"  {escape(last_build_error.strip()[:200])}")

    body = "\n".join(lines).rstrip()
    _c.print(
        Panel(
            body,
            title=title,
            border_style=color,
        )
    )


def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _parse_patch_changes(diff: str) -> list[tuple[str, str]]:
    """Extract (status, patch_filename) for debian/patches/*.patch entries."""
    changes: list[tuple[str, str]] = []
    current: str | None = None
    is_new = is_del = False

    for line in diff.splitlines():
        m = re.match(r'^diff --git a/(.*) b/(.*)$', line)
        if m:
            if current:
                changes.append((_classify(is_new, is_del), current))
            path = m.group(2)
            if "debian/patches/" in path and path.endswith(".patch"):
                current = path.split("/")[-1]
            else:
                current = None
            is_new = is_del = False
            continue
        if current:
            if line.startswith("new file mode"):
                is_new = True
            elif line.startswith("deleted file mode"):
                is_del = True

    if current:
        changes.append((_classify(is_new, is_del), current))

    return changes


def _classify(is_new: bool, is_del: bool) -> str:
    if is_new:
        return "added"
    if is_del:
        return "dropped"
    return "modified"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _show_diff(path: str, old: str, new: str) -> None:
    diff_lines = list(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
    if not diff_lines:
        _c.print(f"  [dim](no change to {escape(path)})[/dim]")
        return
    _c.print(
        Panel(
            Syntax("".join(diff_lines), "diff", word_wrap=True),
            title=f"[bold yellow]edit  {escape(path)}[/bold yellow]",
            border_style="yellow",
        )
    )


def _guess_lang(path: str) -> str:
    if path.endswith(".py"):
        return "python"
    if path.endswith((".yaml", ".yml")):
        return "yaml"
    if path.endswith(".sh"):
        return "bash"
    if path.endswith(".json"):
        return "json"
    return "text"
