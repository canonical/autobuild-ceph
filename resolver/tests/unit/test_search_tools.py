from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ceph_autobuild_resolver.tools.schema import (
    HARD_MAX_MATCHES,
    MAX_EXCERPT_CHARS,
    MAX_FILES_LISTED,
)
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


def test_grep_caps_files_with_matches_and_bounds_payload(fake_lxd, search):
    # Seed far more matching files than the cap. This mirrors the real
    # incident: a broad pattern (e.g. '#include') matched ~every file and the
    # unbounded files_with_matches list overflowed the request token limit.
    n_files = MAX_FILES_LISTED + 50
    for i in range(n_files):
        _seed(fake_lxd, f"debian/f{i:04d}.txt", "needle here\n")
    out = search.grep("needle")

    assert out["total_files"] == n_files
    assert out["files_truncated"] is True
    assert len(out["files_with_matches"]) == MAX_FILES_LISTED
    assert "hint" in out
    # The whole payload must stay well under any sane per-request token limit.
    assert len(json.dumps(out)) < 256 * 1024


def test_grep_clamps_explicit_oversized_max_matches(fake_lxd, search):
    # An EXPLICIT huge max_matches must be clamped (not just the omitted-arg
    # default), or the matches list itself is unbounded.
    body = "\n".join(f"needle {i}" for i in range(HARD_MAX_MATCHES + 100)) + "\n"
    _seed(fake_lxd, "debian/big.txt", body)
    out = search.grep("needle", max_matches=10_000_000)
    assert out["matches_returned"] == HARD_MAX_MATCHES
    assert out["truncated"] is True


def test_grep_max_matches_zero_returns_counts_without_excerpts(fake_lxd, search):
    # Explicit 0 means "no excerpts" -- counts and file list still returned.
    _seed(fake_lxd, "debian/a.txt", "needle\nneedle\n")
    out = search.grep("needle", max_matches=0)
    assert out["matches_returned"] == 0
    assert out["matches"] == []
    assert out["total_matches"] == 2
    assert out["files_with_matches"] == ["debian/a.txt"]
    # The empty page is intentional, not a truncation -- so the model isn't
    # nudged into a pointless paging retry.
    assert out["truncated"] is False


def test_grep_negative_max_matches_clamped_to_zero(fake_lxd, search):
    # A negative value must not produce a matches[offset:offset-n] slice.
    _seed(fake_lxd, "debian/a.txt", "needle\nneedle\nneedle\n")
    out = search.grep("needle", max_matches=-5)
    assert out["matches_returned"] == 0
    assert out["total_matches"] == 3
    assert out["truncated"] is False


def test_grep_truncates_long_excerpt(fake_lxd, search):
    long_line = "needle " + "x" * 5000
    _seed(fake_lxd, "debian/long.txt", long_line + "\n")
    out = search.grep("needle")
    excerpt = out["matches"][0]["excerpt"]
    assert len(excerpt) <= MAX_EXCERPT_CHARS + len(" ...[excerpt truncated]")
    assert excerpt.endswith("...[excerpt truncated]")


def test_grep_not_truncated_when_few_files(fake_lxd, search):
    _seed(fake_lxd, "debian/a.txt", "needle\n")
    _seed(fake_lxd, "debian/b.txt", "needle\n")
    out = search.grep("needle")
    assert out["files_truncated"] is False
    assert out["total_files"] == 2
    assert "hint" not in out


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
