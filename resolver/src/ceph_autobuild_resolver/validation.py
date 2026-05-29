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
    # Derive the validation container name from the pristine container so
    # concurrent resolver runs on different containers don't collide.
    validation_container = f"{pristine_container}-validate"

    # Always start from a clean validation container.
    lxd.delete(validation_container, force=True)
    lxd.copy_from_snapshot(
        pristine_container, pristine_snapshot, validation_container
    )

    try:
        apply_result = runner.apply_diff(validation_container, diff)
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

        outcome = runner.build(validation_container)
        return ValidationResult(ok=outcome.ok, build_outcome=outcome)
    finally:
        # Always delete the temporary validation container so containers don't
        # accumulate across runs. The next run's pre-delete is a safety net
        # only, not the primary cleanup.
        lxd.delete(validation_container, force=True)
