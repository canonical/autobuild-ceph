"""Shared fixtures for resolver tests.

The big shared piece is ``FakeLXD`` — an in-memory stand-in for ``LXDManager``
that mirrors enough of the real interface for the tools and runner to exercise
their full code paths without touching pylxd or LXD itself.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

import pytest

from ceph_autobuild_resolver.config import Config
from ceph_autobuild_resolver.lxd import ExecResult


@dataclass
class FakeLXD:
    """In-memory LXD stand-in with a real on-host scratch tree.

    ``put_text`` / ``push_file`` write into ``root``; ``exec`` executes the
    given argv on the host with ``root`` as the working directory. Any
    container-absolute path in argv (``/root/ceph/...`` etc) is rewritten to
    live under ``root`` so commands like ``cat`` and ``sed`` work as expected.
    """

    root: Path
    workdir_remote: str = "/root/ceph"
    log_remote: str = "/root/build-logs"
    exec_log: list[tuple[str, list[str]]] = field(default_factory=list)
    exec_overrides: dict[str, Callable[[list[str]], ExecResult]] = field(
        default_factory=dict
    )
    files_written: dict[str, str] = field(default_factory=dict)
    # (argv[0], timeout) per exec call, so tests can assert timeout plumbing.
    exec_timeouts: list[tuple[str, int | None]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Path translation
    # ------------------------------------------------------------------

    def _translate(self, path: str) -> Path:
        # Rewrite well-known container roots into our scratch tree.
        for prefix in (self.workdir_remote, self.log_remote, "/tmp", "/root"):
            if path == prefix or path.startswith(prefix + "/"):
                rel = path[len(prefix):].lstrip("/")
                base = self.root / prefix.lstrip("/")
                base.mkdir(parents=True, exist_ok=True)
                return base / rel if rel else base
        # Unknown — keep under root.
        return self.root / path.lstrip("/")

    def _rewrite_argv(self, argv: list[str]) -> list[str]:
        out: list[str] = []
        for a in argv:
            if a.startswith("/") and not a.startswith("-"):
                out.append(str(self._translate(a)))
            else:
                out.append(a)
        return out

    # ------------------------------------------------------------------
    # Real ops
    # ------------------------------------------------------------------

    def push_file(self, local_path: str, container: str, remote_path: str) -> None:
        target = self._translate(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(local_path).read_bytes())
        self.files_written[remote_path] = target.read_text()

    def pull_file(self, container: str, remote_path: str, local_path: str) -> None:
        Path(local_path).write_bytes(self._translate(remote_path).read_bytes())

    def put_text(self, container: str, remote_path: str, content: str) -> None:
        target = self._translate(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        self.files_written[remote_path] = content

    def exec(
        self,
        container: str,
        argv: list[str],
        *,
        env: Mapping[str, str] | None = None,
        check: bool = False,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> ExecResult:
        key = " ".join(argv)
        self.exec_log.append((container, list(argv)))
        self.exec_timeouts.append((argv[0] if argv else "", timeout))
        for prefix, override in self.exec_overrides.items():
            if key.startswith(prefix):
                return override(argv)

        rewritten = self._rewrite_argv(argv)
        host_cwd = str(self._translate(cwd)) if cwd else None
        proc = subprocess.run(
            rewritten, capture_output=True, text=True, check=False, cwd=host_cwd
        )
        # Reverse-translate any host-real paths back to container paths so
        # callers see the same string they'd see in production.
        stdout = self._unrewrite_paths(proc.stdout)
        stderr = self._unrewrite_paths(proc.stderr)
        return ExecResult(proc.returncode, stdout, stderr)

    def _unrewrite_paths(self, text: str) -> str:
        host_root = str(self.root).rstrip("/")
        # Replace ``<root>/<container_path>`` -> ``/<container_path>``.
        return text.replace(host_root + "/", "/")

    def exec_shell(
        self,
        container: str,
        script: str,
        *,
        env: Mapping[str, str] | None = None,
        check: bool = False,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> ExecResult:
        return self.exec(
            container,
            ["bash", "-c", script],
            env=env,
            check=check,
            cwd=cwd,
            timeout=timeout,
        )

    # Lifecycle ops are no-ops in the fake; tests that care override.
    def launch(self, *_args, **_kwargs) -> None:  # pragma: no cover
        pass

    def snapshot(self, *_args, **_kwargs) -> None:  # pragma: no cover
        pass

    def restore_snapshot(self, *_args, **_kwargs) -> None:  # pragma: no cover
        pass

    def copy_from_snapshot(self, *_args, **_kwargs) -> None:  # pragma: no cover
        pass

    def delete(self, *_args, **_kwargs) -> None:  # pragma: no cover
        pass


@pytest.fixture
def cfg() -> Config:
    return Config(
        provider="openrouter",
        model_name="test-model",
        api_key="test-key",
        max_iterations=5,
        max_unchanged_iterations=2,
        run_token_budget=10_000,
        ubuntu_branch="ubuntu/resolute",
        debian_ref="origin/ubuntu/latest",
        launchpad_owner="testuser",
        ceph_version="20.2.0",
    )


@pytest.fixture
def fake_lxd(tmp_path: Path) -> FakeLXD:
    return FakeLXD(root=tmp_path)
