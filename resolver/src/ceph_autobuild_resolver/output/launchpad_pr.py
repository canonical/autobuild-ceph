"""Launchpad PR creation — stub for first integration test.

Until we wire up Launchpad credentials, ``open_pr`` just renders the payload
to stdout (or to disk via ``--dry-run-output`` plus ``--transcript-path``).
The shape of the rendered payload is what the real implementation will send.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass


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
        rendered = {
            "matrix_name": payload.matrix_name,
            "summary": payload.summary,
            "failing_command": payload.failing_command,
            "flags": payload.flags,
            "transcript_path": payload.transcript_path,
            "diff_preview": payload.diff[:2000],
            "diff_bytes": len(payload.diff),
        }
        print("=== resolver: PR payload (dry-run) ===", file=sys.stderr)
        print(json.dumps(rendered, indent=2), file=sys.stderr)
        return
    raise NotImplementedError("Launchpad MR creation not yet wired up")
