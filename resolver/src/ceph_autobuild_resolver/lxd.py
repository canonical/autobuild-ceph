"""LXD container manager built on `pylxd <https://github.com/canonical/pylxd>`_.

Single chokepoint for everything LXD-related so the rest of the resolver
stays testable. Tests inject a fake ``LXDManager`` (or a fake client) instead
of patching subprocess.

Connection model
----------------
``pylxd.Client()`` defaults to the local LXD unix socket. The user must be in
the ``lxd`` group; in CI we make sure that's effective for the resolver step
(``sg lxd -c '...'`` or by running after the membership takes effect).

Why we still expose ``ExecResult``
----------------------------------
``pylxd``'s execute returns a tuple-like with ``exit_code``/``stdout``/``stderr``;
we wrap it in our own dataclass so callers don't depend on pylxd internals
and tests can construct results without importing pylxd at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class LXDError(RuntimeError):
    """Raised when an LXD operation fails for non-build reasons (snapshot
    create errored, container missing, etc). Build-stage failures are
    *expected* and surface as ``ExecResult`` with non-zero ``returncode``,
    not as exceptions."""


class LXDManager:
    """Thin wrapper around a ``pylxd.Client``.

    A ``client`` may be injected for tests. Otherwise we lazy-import pylxd
    on first use so importing this module in unit tests doesn't pull in
    a real LXD dependency.
    """

    def __init__(self, client: Any | None = None) -> None:
        self._client_override = client
        self._client: Any | None = client

    # ------------------------------------------------------------------
    # Client / instance access
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        # Lazy import keeps the test path lightweight.
        import pylxd  # type: ignore[import-not-found]

        try:
            self._client = pylxd.Client()
        except Exception as exc:  # noqa: BLE001
            raise LXDError(f"could not connect to LXD: {exc}") from exc
        return self._client

    def _instance(self, name: str) -> Any:
        try:
            return self._get_client().instances.get(name)
        except Exception as exc:  # noqa: BLE001
            raise LXDError(f"instance {name!r} not found: {exc}") from exc

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def launch(self, image: str, name: str) -> None:
        """Create + start a container from a remote image alias.

        ``image`` may be ``ubuntu:24.04`` style; pylxd accepts the same
        ``server/alias`` syntax that ``lxc launch`` does via the
        ``source.server`` + ``source.alias`` pair below.
        """
        server, alias = _split_image(image)
        config: dict[str, Any] = {
            "name": name,
            "source": {
                "type": "image",
                "mode": "pull",
                "server": server,
                "protocol": "simplestreams",
                "alias": alias,
            },
        }
        client = self._get_client()
        try:
            inst = client.instances.create(config, wait=True)
            inst.start(wait=True)
        except Exception as exc:  # noqa: BLE001
            raise LXDError(f"launch failed for {name!r}: {exc}") from exc
        # cloud-init may still be running; mirror what lxd.sh does.
        result = self.exec(name, ["cloud-init", "status", "--wait"], check=False)
        if not result.ok:
            log.warning("cloud-init wait reported %d", result.returncode)

    def snapshot(self, name: str, snapshot: str) -> None:
        try:
            self._instance(name).snapshots.create(snapshot, wait=True)
        except Exception as exc:  # noqa: BLE001
            raise LXDError(f"snapshot {name}/{snapshot} failed: {exc}") from exc

    def restore_snapshot(self, name: str, snapshot: str) -> None:
        """Roll ``name`` back to a previously-created snapshot.

        LXD restores in place: any state accumulated since the snapshot was
        taken (filesystem changes, processes, etc.) is discarded. Restore
        may leave the container stopped, so we re-start it if needed.
        """
        try:
            self._instance(name).restore_snapshot(snapshot, wait=True)
            inst = self._instance(name)
            # status_code 103 = RUNNING in LXD's REST API.
            if inst.status_code != 103:
                inst.start(wait=True)
        except Exception as exc:  # noqa: BLE001
            raise LXDError(
                f"restore {name}/{snapshot} failed: {exc}"
            ) from exc

    def copy_from_snapshot(self, src: str, snapshot: str, dst: str) -> None:
        """Materialise ``src/snapshot`` as a fresh, started container ``dst``."""
        client = self._get_client()
        config: dict[str, Any] = {
            "name": dst,
            "source": {
                "type": "copy",
                "source": f"{src}/{snapshot}",
            },
        }
        try:
            inst = client.instances.create(config, wait=True)
            inst.start(wait=True)
        except Exception as exc:  # noqa: BLE001
            raise LXDError(
                f"copy {src}/{snapshot} -> {dst} failed: {exc}"
            ) from exc

    def attach_disk_device(
        self,
        container: str,
        device_name: str,
        host_path: str,
        container_path: str,
    ) -> None:
        """Bind-mount ``host_path`` into ``container`` at ``container_path``.

        Idempotent: if the device already exists with the same source and path,
        this is a no-op.  Replaces the device if only the paths changed.
        """
        try:
            inst = self._instance(container)
            existing = inst.devices.get(device_name, {})
            if existing.get("source") == host_path and existing.get("path") == container_path:
                return
            inst.devices[device_name] = {
                "type": "disk",
                "source": host_path,
                "path": container_path,
            }
            inst.save(wait=True)
        except Exception as exc:  # noqa: BLE001
            raise LXDError(
                f"attach_disk_device {device_name!r} on {container!r} failed: {exc}"
            ) from exc

    def delete(self, name: str, force: bool = True) -> None:
        try:
            inst = self._get_client().instances.get(name)
        except Exception:  # noqa: BLE001
            return  # already gone, nothing to do
        try:
            if force and inst.status_code != 102:  # 102 = STOPPED
                inst.stop(force=True, wait=True)
            inst.delete(wait=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("delete %s failed: %s", name, exc)

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    def push_file(self, local_path: str, container: str, remote_path: str) -> None:
        with open(local_path, "rb") as fh:
            data = fh.read()
        self._instance(container).files.put(remote_path, data)

    def pull_file(self, container: str, remote_path: str, local_path: str) -> None:
        data = self._instance(container).files.get(remote_path)
        with open(local_path, "wb") as fh:
            fh.write(data)

    def put_text(self, container: str, remote_path: str, content: str) -> None:
        """Write a string to a file inside the container.

        Used by the model's ``write_file`` / ``edit_file`` tools — much
        cleaner than the base64-via-shell dance we'd need with raw ``lxc``.
        """
        self._instance(container).files.put(
            remote_path, content.encode("utf-8")
        )

    # ------------------------------------------------------------------
    # Exec
    # ------------------------------------------------------------------

    def exec(
        self,
        container: str,
        argv: list[str],
        *,
        env: Mapping[str, str] | None = None,
        check: bool = False,
        cwd: str | None = None,
    ) -> ExecResult:
        try:
            raw = self._instance(container).execute(
                argv,
                environment=dict(env) if env else None,
                cwd=cwd,
            )
        except Exception as exc:  # noqa: BLE001
            raise LXDError(f"exec on {container} failed to dispatch: {exc}") from exc

        result = ExecResult(
            returncode=int(raw.exit_code),
            stdout=raw.stdout or "",
            stderr=raw.stderr or "",
        )
        if check and not result.ok:
            raise LXDError(
                f"command failed ({result.returncode}) on {container}: "
                f"{argv}\nstderr: {result.stderr}"
            )
        return result

    def exec_shell(
        self,
        container: str,
        script: str,
        *,
        env: Mapping[str, str] | None = None,
        check: bool = False,
        cwd: str | None = None,
    ) -> ExecResult:
        return self.exec(
            container, ["bash", "-c", script], env=env, check=check, cwd=cwd
        )


def _split_image(image: str) -> tuple[str, str]:
    """Split ``server:alias`` into a ``(server_url, alias)`` pair.

    Matches what ``lxc launch`` accepts. ``ubuntu:24.04`` -> the ubuntu
    simplestreams server; ``ubuntu-daily:resolute`` -> the daily one.
    Defaults to the canonical server when no prefix is given.
    """
    known = {
        "ubuntu": "https://cloud-images.ubuntu.com/releases",
        "ubuntu-daily": "https://cloud-images.ubuntu.com/daily",
        "images": "https://images.linuxcontainers.org",
    }
    if ":" in image:
        prefix, alias = image.split(":", 1)
        return known.get(prefix, prefix), alias
    return known["ubuntu"], image
