"""Clean-rebuild validation.

After the model declares a fix, we copy the pristine snapshot into a fresh
container, apply the proposed diff (and *only* that diff), and run the build
stage. If it succeeds, we have a validated fix — patches that only build
under accumulated iteration-container state will fail this step, which is
exactly the guard the spec calls for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .build_runner import BuildOutcome, BuildRunner
from .lxd import LXDManager

log = logging.getLogger(__name__)

VALIDATION_CONTAINER = "ceph-build-validate"


@dataclass
class ValidationResult:
    ok: bool
    build_outcome: BuildOutcome
    apply_stderr: str = ""


def validate(
    *,
    lxd: LXDManager,
    runner: BuildRunner,
    pristine_container: str,
    pristine_snapshot: str,
    diff: str,
) -> ValidationResult:
    # Always start from a clean validation container.
    lxd.delete(VALIDATION_CONTAINER, force=True)
    lxd.copy_from_snapshot(
        pristine_container, pristine_snapshot, VALIDATION_CONTAINER
    )

    apply_result = runner.apply_diff(VALIDATION_CONTAINER, diff)
    if not apply_result.ok:
        # Don't even bother building; the diff itself is malformed.
        return ValidationResult(
            ok=False,
            build_outcome=BuildOutcome(
                ok=False,
                stage="apply_diff",
                returncode=apply_result.returncode,
                log_path="",
                log_tail="",
            ),
            apply_stderr=apply_result.stderr,
        )

    outcome = runner.build(VALIDATION_CONTAINER)
    return ValidationResult(ok=outcome.ok, build_outcome=outcome)
