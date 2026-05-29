"""Launchpad MR creation.

Launchpad API integration is not yet wired up. Both dry_run=True and
dry_run=False currently route through ci_output.emit_success so the resolver
produces CI-friendly output in all cases. When real integration is added,
the dry_run=False branch should call the Launchpad API instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .ci_output import CISuccessPayload, emit_success

log = logging.getLogger(__name__)


@dataclass
class PRPayload:
    matrix_name: str
    summary: str
    diff: str
    transcript_path: str


def open_pr(payload: PRPayload, dry_run: bool = True) -> None:
    if not dry_run:
        # Launchpad integration not yet wired up — fall through to CI output
        # so a successful resolution is not lost in an unhandled exception.
        log.warning("Launchpad MR creation not implemented; emitting CI output instead")
    emit_success(CISuccessPayload(
        matrix_name=payload.matrix_name,
        summary=payload.summary,
        diff=payload.diff,
        transcript_path=payload.transcript_path,
    ))
