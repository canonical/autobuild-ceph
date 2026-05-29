"""Launchpad bug filing.

Launchpad API integration is not yet wired up. Both dry_run=True and
dry_run=False currently route through ci_output.emit_failure so the resolver
produces CI-friendly failure analysis in all cases. When real integration is
added, the dry_run=False branch should call the Launchpad API instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .ci_output import CIFailurePayload, emit_failure

log = logging.getLogger(__name__)


@dataclass
class BugPayload:
    matrix_name: str
    failing_command: str
    error_tail: str
    transcript_path: str
    stop_reason: str


def file_bug(payload: BugPayload, dry_run: bool = True) -> None:
    if not dry_run:
        # Launchpad integration not yet wired up — fall through to CI output
        # so failure details are not lost in an unhandled exception.
        log.warning("Launchpad bug filing not implemented; emitting CI output instead")
    emit_failure(CIFailurePayload(
        matrix_name=payload.matrix_name,
        stop_reason=payload.stop_reason,
        failing_command=payload.failing_command,
        error_tail=payload.error_tail,
        transcript_path=payload.transcript_path,
    ))
