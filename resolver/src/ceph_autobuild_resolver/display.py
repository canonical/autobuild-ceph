"""Terminal display helpers for the resolver UI.

All output goes to stderr so stdout stays clean for any machine-readable
output.  Rich auto-detects whether stderr is a TTY; in non-interactive
contexts (CI, unit tests with captured stderr) panels still render but
without ANSI colour — which is fine.
"""

from __future__ import annotations

import difflib

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

_c = Console(stderr=True, highlight=False)


# ---------------------------------------------------------------------------
# Prep / build-runner output
# ---------------------------------------------------------------------------


def stage_header(name: str) -> None:
    _c.rule(f"[bold white]{name}[/bold white]")


def step_start(argv: list[str]) -> None:
    label = " ".join(str(a) for a in argv[:3])
    if len(argv) > 3:
        label += " …"
    _c.print(f"  [dim cyan]→ {label}[/dim cyan]")


def step_done(ok: bool, returncode: int = 0, output: str = "") -> None:
    if ok:
        _c.print("    [bold green]✓[/bold green]")
    else:
        _c.print(f"    [bold red]✗  rc={returncode}[/bold red]")
        tail = output.strip()[-2000:] if output.strip() else ""
        if tail:
            _c.print(tail, style="dim red")


# ---------------------------------------------------------------------------
# Resolve-loop output
# ---------------------------------------------------------------------------


def model_text(text: str) -> None:
    if not text.strip():
        return
    _c.print(
        Panel(
            text.strip(),
            title="[bold cyan]model[/bold cyan]",
            border_style="cyan",
        )
    )


def model_reasoning(text: str) -> None:
    if not text.strip():
        return
    _c.print(
        Panel(
            text.strip(),
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
        f"[dim]{k}[/dim]=[cyan]{repr(v)[:80]}[/cyan]"
        for k, v in list(args.items())[:4]
    )
    suffix = " …" if len(args) > 4 else ""
    _c.print(f"  [bold blue]⚙ {name}[/bold blue]({parts}{suffix})")


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
                title=f"[bold green]new file  {path}[/bold green]",
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
            report,
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
            tail,
            title=f"[bold {color}]build {status} @ {stage}[/bold {color}]",
            border_style=color,
        )
    )


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
        _c.print(f"  [dim](no change to {path})[/dim]")
        return
    _c.print(
        Panel(
            Syntax("".join(diff_lines), "diff", word_wrap=True),
            title=f"[bold yellow]edit  {path}[/bold yellow]",
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
