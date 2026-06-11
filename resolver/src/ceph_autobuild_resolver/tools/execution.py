"""Execution-side tool handlers (run_build, clean, check_patch)."""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass
from typing import Any

from .. import guards
from ..build_runner import BuildOutcome, BuildRunner
from ..config import TOOL_EXEC_TIMEOUT_SECONDS

log = logging.getLogger(__name__)


@dataclass
class ExecutionHandlers:
    runner: BuildRunner
    container: str
    last_build: BuildOutcome | None = None
    # Set by Dispatcher whenever a file-mutating tool succeeds; cleared
    # whenever run_build actually executes. The hard guard below uses
    # this to refuse pointless rebuilds.
    files_changed_since_last_build: bool = False

    def run_build(self) -> dict[str, Any]:
        # Hard guard: if the previous build failed and nothing has changed
        # since, there's no reason to re-run. Refuse with a structured
        # error so the model can self-correct on the next turn.
        if (
            self.last_build is not None
            and not self.last_build.ok
            and not self.files_changed_since_last_build
        ):
            return {
                "ok": False,
                "skipped": True,
                "error": "no_changes_since_last_build",
                "message": (
                    "Refusing to re-run the build: nothing has changed since "
                    f"the previous failed build (stage={self.last_build.stage!r}, "
                    f"rc={self.last_build.returncode}). The result will be "
                    "identical. Use grep_log/read_log to locate the root "
                    "cause, then edit a file before calling run_build again."
                ),
                "stage": self.last_build.stage,
                "returncode": self.last_build.returncode,
                "log_path": self.last_build.log_path,
            }
        outcome = self.runner.build(self.container)
        self.last_build = outcome
        self.files_changed_since_last_build = False
        return {
            "ok": outcome.ok,
            "stage": outcome.stage,
            "returncode": outcome.returncode,
            "log_path": outcome.log_path,
            "log_tail": outcome.log_tail,
        }

    def check_patch(self, path: str) -> dict[str, Any]:
        """Dry-run a patch using the same flags dpkg-source uses.

        Runs ``patch -F 0 -p1 --dry-run`` against the working tree *in its
        current state* — whatever state quilt has left it in. The caller is
        responsible for ensuring the preceding patches in series are already
        applied if they want a faithful simulation of series-order application.
        """
        guards.assert_in_scope(path)
        rel = guards.normalize(path)
        full = f"{self.runner.cfg.container_workdir}/{rel}"
        workdir = self.runner.cfg.container_workdir
        cmd = f"patch -F 0 -p1 --dry-run < {shlex.quote(full)}"
        result = self.runner.lxd.exec(
            self.container,
            ["bash", "-c", cmd],
            cwd=workdir,
            check=False,
            timeout=TOOL_EXEC_TIMEOUT_SECONDS,
        )
        if result.timed_out:
            return {
                "ok": False,
                "output": (
                    f"check_patch exceeded {TOOL_EXEC_TIMEOUT_SECONDS}s and "
                    "was killed."
                ),
            }
        return {
            "ok": result.ok,
            "output": (result.stdout + result.stderr).strip(),
        }

    def clean(self) -> dict[str, Any]:
        # ``debuild`` leaves *.buildinfo / *.changes / *.deb at the parent of
        # the source tree. container_workdir is always /root/ceph, so ``/..``
        # resolves to /root where debuild drops its output.
        result = self.runner.lxd.exec_shell(
            self.container,
            f"cd {self.runner.cfg.container_workdir}/.. && rm -f *.buildinfo *.changes *.deb",
            check=False,
            timeout=TOOL_EXEC_TIMEOUT_SECONDS,
        )
        return {"ok": result.ok}
