from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ceph_autobuild_resolver.tools.search import SearchHandlers


@pytest.fixture
def search(fake_lxd, cfg) -> SearchHandlers:
    return SearchHandlers(lxd=fake_lxd, cfg=cfg, container="ceph-build")


def _seed(fake_lxd, rel: str, content: str) -> None:
    full = Path(fake_lxd.root) / "root/ceph" / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content)


def test_grep_returns_matches_and_files_with_matches(fake_lxd, search):
    _seed(fake_lxd, "debian/a.txt", "alpha TODO\nbeta\n")
    _seed(fake_lxd, "debian/b.txt", "gamma\nTODO again\n")
    out = search.grep("TODO")
    assert out["total_matches"] == 2
    assert sorted(out["files_with_matches"]) == [
        "debian/a.txt",
        "debian/b.txt",
    ]
    assert out["truncated"] is False


def test_grep_paginates(fake_lxd, search):
    body = "\n".join(f"TODO line {i}" for i in range(60)) + "\n"
    _seed(fake_lxd, "debian/big.txt", body)
    out = search.grep("TODO", max_matches=50)
    assert out["matches_returned"] == 50
    assert out["truncated"] is True
    out2 = search.grep("TODO", max_matches=50, match_offset=50)
    assert out2["matches_returned"] == 10
    assert out2["truncated"] is False


def test_git_log_handles_non_repo(fake_lxd, search):
    # No git init in the scratch tree -> git log fails.
    out = search.git_log()
    assert "error" in out


def test_git_log_returns_commits(fake_lxd, cfg):
    workdir = Path(fake_lxd.root) / "root/ceph"
    workdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=workdir, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=T", "commit",
         "--allow-empty", "-m", "first"],
        cwd=workdir, check=True,
    )
    s = SearchHandlers(lxd=fake_lxd, cfg=cfg, container="ceph-build")
    out = s.git_log(n=5)
    assert "commits" in out
    assert len(out["commits"]) == 1
    assert out["commits"][0]["subject"] == "first"
