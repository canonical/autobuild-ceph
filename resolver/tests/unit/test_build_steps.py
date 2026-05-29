"""Sanity checks for build_steps stage builders."""

from __future__ import annotations

import dataclasses

import pytest

from ceph_autobuild_resolver.build_steps import (
    _debuild_cmd,
    build_stage,
    install_build_requirements_stage,
    install_dependencies_stage,
    prepare_tarball_stage,
)
from ceph_autobuild_resolver.config import Config


@pytest.fixture
def _cfg(cfg) -> Config:
    return cfg


def _flat(stage) -> list[str]:
    """Flatten all argv entries from a stage into one list for easy searching.

    bash -c '...' steps are split by whitespace so their embedded commands are
    searchable the same way as plain argv entries.
    """
    result = []
    for step in stage.steps:
        for token in step.argv:
            result.extend(token.split())
    return result


def test_install_dependencies_stage_non_empty(_cfg):
    stage = install_dependencies_stage(_cfg)
    assert stage.name == "install_dependencies"
    assert len(stage.steps) > 0


def test_install_dependencies_stage_has_apt(_cfg):
    flat = _flat(install_dependencies_stage(_cfg))
    assert "apt" in flat
    assert "devscripts" in flat
    assert "git-buildpackage" in flat


def test_prepare_tarball_stage_non_empty(_cfg):
    stage = prepare_tarball_stage(_cfg)
    assert stage.name == "prepare_tarball"
    assert len(stage.steps) > 0


def test_prepare_tarball_stage_clones_ceph(_cfg):
    flat = _flat(prepare_tarball_stage(_cfg))
    assert "https://github.com/ceph/ceph" in flat


def test_prepare_tarball_stage_import_orig(_cfg):
    flat = _flat(prepare_tarball_stage(_cfg))
    assert "import-orig" in flat


def test_prepare_tarball_stage_final_commit(_cfg):
    stage = prepare_tarball_stage(_cfg)
    last = stage.steps[-1]
    assert "git" in last.argv
    assert "commit" in last.argv


def test_prepare_tarball_stage_embeds_config(_cfg):
    stage = prepare_tarball_stage(_cfg)
    flat = _flat(stage)
    assert f"v{_cfg.ceph_version}" in flat
    assert _cfg.ubuntu_branch in flat
    assert _cfg.launchpad_owner in " ".join(flat)


def test_install_build_requirements_stage_non_empty(_cfg):
    stage = install_build_requirements_stage(_cfg)
    assert stage.name == "install_build_requirements"
    assert len(stage.steps) > 0


def test_install_build_requirements_stage_has_mk_build_deps(_cfg):
    flat = _flat(install_build_requirements_stage(_cfg))
    assert "mk-build-deps" in flat


def test_build_stage_non_empty(_cfg):
    stage = build_stage(_cfg)
    assert stage.name == "build"
    assert len(stage.steps) > 0


def test_build_stage_has_debuild(_cfg):
    flat = _flat(build_stage(_cfg))
    assert "debuild" in flat


def test_debuild_cmd_no_ccache_when_disabled(_cfg):
    # The default fixture has ccache_host_dir=None.
    assert _cfg.ccache_host_dir is None
    cmd = _debuild_cmd(_cfg)
    assert "ccache" not in cmd
    assert "--prepend-path" not in cmd
    assert cmd == "debuild --no-lintian -us -uc -d -b -j$(nproc)"


def test_debuild_cmd_uses_debuild_knobs_for_ccache(_cfg):
    cfg = dataclasses.replace(_cfg, ccache_host_dir="/var/cache/ccache/ceph-resolute")
    cmd = _debuild_cmd(cfg)
    # debuild's own mechanisms, not a shell PATH/env prefix it would scrub.
    assert "debuild --prepend-path=/usr/lib/ccache" in cmd
    assert "--preserve-envvar='CCACHE_*'" in cmd
    assert "CCACHE_DIR=/root/ccache" in cmd
    assert f"CCACHE_BASEDIR={cfg.container_workdir}" in cmd
    # The old broken form must be gone (debuild resets PATH, so this was a no-op).
    assert "PATH=/usr/lib/ccache:$PATH" not in cmd


def test_debuild_cmd_ccache_max_size_is_configurable(_cfg):
    cfg = dataclasses.replace(
        _cfg, ccache_host_dir="/var/cache/ccache/x", ccache_max_size="7G"
    )
    assert "CCACHE_MAXSIZE=7G" in _debuild_cmd(cfg)
