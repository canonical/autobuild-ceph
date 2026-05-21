"""Launchpad MR creation.

dry_run=True (the default when --dry-run-output is passed) routes through
ci_output.emit_success so the resolver produces CI-friendly output without
requiring Launchpad credentials.

dry_run=False will submit to Launchpad once that integration is wired up.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ci_output import CISuccessPayload, emit_success


@dataclass
class PRPayload:
    matrix_name: str
    summary: str
    diff: str
    failing_command: str
    original_error_tail: str
    transcript_path: str
    flags: dict[str, bool]


def open_pr(payload: PRPayload, dry_run: bool = True) -> None:
    if dry_run:
        emit_success(CISuccessPayload(
            matrix_name=payload.matrix_name,
            summary=payload.summary,
            diff=payload.diff,
            transcript_path=payload.transcript_path,
        ))
        return
    raise NotImplementedError("Launchpad MR creation not yet wired up")
