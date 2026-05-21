"""Launchpad bug filing.

dry_run=True routes through ci_output.emit_failure so the resolver produces
CI-friendly failure analysis without requiring Launchpad credentials.

dry_run=False will file a Launchpad bug once that integration is wired up.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ci_output import CIFailurePayload, emit_failure


@dataclass
class BugPayload:
    matrix_name: str
    failing_command: str
    error_tail: str
    transcript_path: str
    stop_reason: str


def file_bug(payload: BugPayload, dry_run: bool = True) -> None:
    if dry_run:
        emit_failure(CIFailurePayload(
            matrix_name=payload.matrix_name,
            stop_reason=payload.stop_reason,
            failing_command=payload.failing_command,
            error_tail=payload.error_tail,
            transcript_path=payload.transcript_path,
        ))
        return
    raise NotImplementedError("Launchpad bug creation not yet wired up")
