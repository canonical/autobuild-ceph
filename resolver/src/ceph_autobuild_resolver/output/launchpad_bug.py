"""Launchpad bug filing — stub. Mirrors launchpad_pr.py."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass


@dataclass
class BugPayload:
    matrix_name: str
    failing_command: str
    error_tail: str
    transcript_path: str
    stop_reason: str


def file_bug(payload: BugPayload, dry_run: bool = True) -> None:
    if dry_run:
        rendered = {
            "matrix_name": payload.matrix_name,
            "failing_command": payload.failing_command,
            "stop_reason": payload.stop_reason,
            "transcript_path": payload.transcript_path,
            "error_tail_preview": payload.error_tail[-2000:],
        }
        print("=== resolver: bug payload (dry-run) ===", file=sys.stderr)
        print(json.dumps(rendered, indent=2), file=sys.stderr)
        return
    raise NotImplementedError("Launchpad bug creation not yet wired up")
