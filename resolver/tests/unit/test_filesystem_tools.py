"""Tests for the filesystem tool handlers using the FakeLXD fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from ceph_autobuild_resolver.build_runner import BuildRunner
from ceph_autobuild_resolver import guards
from ceph_autobuild_resolver.tools.filesystem import (
    FilesystemHandlers,
    _diff_target_paths,
)


@pytest.fixture
def fs(fake_lxd, cfg) -> FilesystemHandlers:
    runner = BuildRunner(fake_lxd, cfg)
    return FilesystemHandlers(
        lxd=fake_lxd, cfg=cfg, container="ceph-build", runner=runner
    )


def _seed_file(fake_lxd, rel_path: str, content: str) -> None:
    full = Path(fake_lxd.root) / "root/ceph" / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content)


def test_read_files_returns_full_when_under_cap(fake_lxd, fs):
    _seed_file(fake_lxd, "debian/control", "Source: ceph\n")
    out = fs.read_files(["debian/control"])
    assert len(out["files"]) == 1
    f = out["files"][0]
    assert f["content"] == "Source: ceph\n"
    assert f["truncated"] is False
    assert f["total_lines"] == 1


def test_read_files_truncates_with_metadata(fake_lxd, fs):
    content = "\n".join(f"line {i}" for i in range(1, 1001)) + "\n"
    _seed_file(fake_lxd, "debian/big.txt", content)
    out = fs.read_files(["debian/big.txt"])
    f = out["files"][0]
    assert f["truncated"] is True
    assert f["total_lines"] == 1000
    assert f["start_line"] == 1
    assert f["end_line"] == 500
    # Follow-up call requesting next slice
    out2 = fs.read_files(["debian/big.txt"], start_line=501, end_line=1000)
    f2 = out2["files"][0]
    assert f2["truncated"] is False
    assert f2["start_line"] == 501


def test_read_files_handles_missing_file(fake_lxd, fs):
    out = fs.read_files(["debian/does-not-exist"])
    assert "error" in out["files"][0]


def test_write_file_rejects_upstream_path(fs):
    with pytest.raises(Exception) as exc:
        fs.write_file("src/common/buffer.cc", "x")
    assert "outside debian/" in str(exc.value)


def test_write_file_succeeds_under_debian(fake_lxd, fs):
    # Non-patch debian files are allowed.
    out = fs.write_file("debian/rules", "content")
    assert out["ok"] is True
    written = (Path(fake_lxd.root) / "root/ceph/debian/rules").read_text()
    assert written == "content"


def test_write_file_blocks_patch_files(fake_lxd, fs):
    with pytest.raises(guards.EditScopeViolation):
        fs.write_file("debian/patches/new.patch", "content")


def test_write_file_flags_series_removal(fake_lxd, fs):
    _seed_file(fake_lxd, "debian/patches/series", "p1.patch\np2.patch\n")
    out = fs.write_file("debian/patches/series", "p1.patch\n")
    assert out["ok"] is True
    assert out["flags"]["debian_patches_series_removal"] is True


def test_edit_file_requires_unique_match(fake_lxd, fs):
    _seed_file(fake_lxd, "debian/control", "x\nx\n")
    out = fs.edit_file("debian/control", "x\n", "y\n")
    assert out["ok"] is False
    assert "more than once" in out["error"] or "matches" in out["error"]


def test_edit_file_replaces_unique_match(fake_lxd, fs):
    _seed_file(fake_lxd, "debian/control", "Source: ceph\nVersion: old\n")
    out = fs.edit_file("debian/control", "Version: old", "Version: new")
    assert out["ok"] is True
    assert (
        Path(fake_lxd.root) / "root/ceph/debian/control"
    ).read_text() == "Source: ceph\nVersion: new\n"


def test_edit_file_missing_old_str(fake_lxd, fs):
    _seed_file(fake_lxd, "debian/control", "x\n")
    out = fs.edit_file("debian/control", "missing", "y")
    assert out["ok"] is False
    assert "not found" in out["error"]


def test_diff_target_paths_extracts_b_targets():
    diff = (
        "diff --git a/debian/patches/x.patch b/debian/patches/x.patch\n"
        "--- a/debian/patches/x.patch\n"
        "+++ b/debian/patches/x.patch\n"
        "@@ ...\n"
    )
    assert _diff_target_paths(diff) == ["debian/patches/x.patch"]


def test_apply_patch_rejects_upstream_targets(fs):
    diff = (
        "--- a/src/foo.cc\n"
        "+++ b/src/foo.cc\n"
        "@@ -1 +1 @@\n"
        "-x\n+y\n"
    )
    with pytest.raises(Exception):
        fs.apply_patch(diff)


# ----------------------------------------------------------------------
# replace_in_upstream / drop_patch
# ----------------------------------------------------------------------


def test_replace_in_upstream_generates_well_formed_patch(fake_lxd, fs):
    upstream = "line a\nline b\nline c\nline d\nline e\n"
    _seed_file(fake_lxd, "src/foo.cc", upstream)
    _seed_file(fake_lxd, "debian/patches/series", "")

    out = fs.replace_in_upstream(
        patch_name="fix-foo.patch",
        file="src/foo.cc",
        old_content="line c\n",
        new_content="line C MODIFIED\n",
        description="Test fix",
    )
    assert out["ok"] is True
    assert out["patch_path"] == "debian/patches/fix-foo.patch"
    assert out["appended_to_series"] is True

    patch = (
        Path(fake_lxd.root) / "root/ceph/debian/patches/fix-foo.patch"
    ).read_text()
    # DEP-3 headers
    assert patch.startswith("Description: Test fix\n")
    assert "Author: ceph-autobuild-resolver" in patch
    assert "Forwarded: not-needed" in patch
    # Standard unified diff form
    assert "--- a/src/foo.cc" in patch
    assert "+++ b/src/foo.cc" in patch
    assert "@@ " in patch
    assert "-line c" in patch
    assert "+line C MODIFIED" in patch
    # Series updated
    series = (
        Path(fake_lxd.root) / "root/ceph/debian/patches/series"
    ).read_text()
    assert "fix-foo.patch" in series


def test_replace_in_upstream_rejects_non_unique_match(fake_lxd, fs):
    _seed_file(fake_lxd, "src/foo.cc", "x\nx\n")
    out = fs.replace_in_upstream(
        patch_name="p.patch",
        file="src/foo.cc",
        old_content="x\n",
        new_content="y\n",
        description="d",
    )
    assert out["ok"] is False
    assert "matches" in out["error"]


def test_replace_in_upstream_rejects_missing_old(fake_lxd, fs):
    _seed_file(fake_lxd, "src/foo.cc", "abc\n")
    out = fs.replace_in_upstream(
        patch_name="p.patch",
        file="src/foo.cc",
        old_content="zzz",
        new_content="yyy",
        description="d",
    )
    assert out["ok"] is False
    assert "not found" in out["error"]


def test_replace_in_upstream_rejects_bad_patch_name(fake_lxd, fs):
    _seed_file(fake_lxd, "src/foo.cc", "abc\n")
    out = fs.replace_in_upstream(
        patch_name="not-ending-in-extension",
        file="src/foo.cc",
        old_content="abc",
        new_content="def",
        description="d",
    )
    assert out["ok"] is False
    assert "patch_name" in out["error"]


def test_replace_in_upstream_appends_diff_for_second_file(fake_lxd, fs):
    """Second call with the same patch_name appends rather than overwrites."""
    _seed_file(fake_lxd, "src/foo.cc", "line a\nline b\nline c\n")
    _seed_file(fake_lxd, "src/bar.cc", "line x\nline y\nline z\n")
    _seed_file(fake_lxd, "debian/patches/series", "")

    fs.replace_in_upstream(
        patch_name="multi.patch",
        file="src/foo.cc",
        old_content="line b\n",
        new_content="line B\n",
        description="Multi-file fix",
    )
    fs.replace_in_upstream(
        patch_name="multi.patch",
        file="src/bar.cc",
        old_content="line y\n",
        new_content="line Y\n",
        description="Multi-file fix",
    )

    patch = (
        Path(fake_lxd.root) / "root/ceph/debian/patches/multi.patch"
    ).read_text()

    # DEP-3 headers appear exactly once.
    assert patch.count("Description: Multi-file fix") == 1
    # Both files are present in the patch.
    assert "--- a/src/foo.cc" in patch
    assert "--- a/src/bar.cc" in patch
    assert "-line b" in patch
    assert "+line B" in patch
    assert "-line y" in patch
    assert "+line Y" in patch
    # Only one series entry.
    series = (
        Path(fake_lxd.root) / "root/ceph/debian/patches/series"
    ).read_text()
    assert series.count("multi.patch") == 1


def test_replace_in_upstream_does_not_double_append(fake_lxd, fs):
    _seed_file(fake_lxd, "src/foo.cc", "abc\n")
    _seed_file(fake_lxd, "debian/patches/series", "fix-foo.patch\n")
    out = fs.replace_in_upstream(
        patch_name="fix-foo.patch",
        file="src/foo.cc",
        old_content="abc",
        new_content="def",
        description="d",
    )
    assert out["ok"] is True
    assert out["appended_to_series"] is False
    series = (
        Path(fake_lxd.root) / "root/ceph/debian/patches/series"
    ).read_text()
    # Only one occurrence
    assert series.count("fix-foo.patch") == 1


def test_drop_patch_removes_series_entry_and_file(fake_lxd, fs):
    _seed_file(fake_lxd, "debian/patches/series", "a.patch\nb.patch\nc.patch\n")
    _seed_file(fake_lxd, "debian/patches/b.patch", "patch contents")

    out = fs.drop_patch("b.patch")
    assert out["ok"] is True
    assert out["removed_from_series"] is True
    assert out["deleted_file"] is True

    series = (
        Path(fake_lxd.root) / "root/ceph/debian/patches/series"
    ).read_text()
    assert series == "a.patch\nc.patch\n"
    assert not (
        Path(fake_lxd.root) / "root/ceph/debian/patches/b.patch"
    ).exists()


def test_drop_patch_when_only_series_has_entry(fake_lxd, fs):
    _seed_file(fake_lxd, "debian/patches/series", "a.patch\nghost.patch\n")
    out = fs.drop_patch("ghost.patch")
    assert out["ok"] is True
    assert out["removed_from_series"] is True
    assert out["deleted_file"] is False
    series = (
        Path(fake_lxd.root) / "root/ceph/debian/patches/series"
    ).read_text()
    assert "ghost" not in series
