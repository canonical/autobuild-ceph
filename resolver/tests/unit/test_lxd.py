"""LXDManager exec: per-command timeout plumbing."""

from __future__ import annotations

from types import SimpleNamespace

from ceph_autobuild_resolver.lxd import LXDManager


class _FakeInstance:
    def __init__(self, exit_code: int = 0):
        self.exit_code = exit_code
        self.calls: list[list[str]] = []

    def execute(self, argv, environment=None, cwd=None):
        self.calls.append(list(argv))
        return SimpleNamespace(exit_code=self.exit_code, stdout="", stderr="")


def _manager(inst: _FakeInstance) -> LXDManager:
    client = SimpleNamespace(instances=SimpleNamespace(get=lambda name: inst))
    return LXDManager(client=client)


def test_exec_wraps_argv_with_coreutils_timeout():
    inst = _FakeInstance(exit_code=0)
    mgr = _manager(inst)
    mgr.exec("c", ["sleep", "1"], timeout=5)
    assert inst.calls[0][:4] == ["timeout", "-k", "10", "5"]
    assert inst.calls[0][4:] == ["sleep", "1"]


def test_exec_maps_expiry_to_timed_out():
    inst = _FakeInstance(exit_code=124)
    res = _manager(inst).exec("c", ["sleep", "999"], timeout=5)
    assert res.timed_out is True
    assert res.ok is False


def test_exec_kill_after_grace_also_timed_out():
    inst = _FakeInstance(exit_code=137)
    res = _manager(inst).exec("c", ["sleep", "999"], timeout=5)
    assert res.timed_out is True


def test_exec_without_timeout_is_unwrapped_and_never_timed_out():
    inst = _FakeInstance(exit_code=124)
    res = _manager(inst).exec("c", ["false"])
    assert inst.calls[0] == ["false"]
    # rc 124 from the command itself is not a timeout when none was requested.
    assert res.timed_out is False


def test_exec_shell_forwards_timeout():
    inst = _FakeInstance(exit_code=0)
    _manager(inst).exec_shell("c", "echo hi", timeout=7)
    assert inst.calls[0][:4] == ["timeout", "-k", "10", "7"]
    assert inst.calls[0][4:6] == ["bash", "-c"]
