from __future__ import annotations

import pytest

from ceph_autobuild_resolver import guards


def test_assert_in_scope_accepts_debian_paths():
    guards.assert_in_scope("debian/control")
    guards.assert_in_scope("debian/patches/series")
    guards.assert_in_scope("./debian/changelog")


def test_assert_in_scope_rejects_upstream_paths():
    with pytest.raises(guards.EditScopeViolation):
        guards.assert_in_scope("src/common/buffer.cc")


def test_assert_in_scope_rejects_script_files_in_debian():
    for ext in (".py", ".sh", ".rb", ".pl"):
        with pytest.raises(guards.EditScopeViolation, match="script"):
            guards.assert_in_scope(f"debian/helper{ext}")


def test_assert_in_scope_rejects_script_files_outside_debian():
    with pytest.raises(guards.EditScopeViolation):
        guards.assert_in_scope("make_patch.py")


def test_assert_in_scope_accepts_patch_and_rules():
    guards.assert_in_scope("debian/patches/fix-foo.patch")
    guards.assert_in_scope("debian/rules")
    guards.assert_in_scope("debian/control")


def test_assert_in_scope_rejects_traversal():
    with pytest.raises(guards.EditScopeViolation):
        guards.assert_in_scope("debian/../etc/passwd")


def test_evaluate_flags_detects_series_removal():
    old = "patch-a.patch\npatch-b.patch\npatch-c.patch\n"
    new = "patch-a.patch\npatch-c.patch\n"
    flags = guards.evaluate_flags(
        "debian/patches/series",
        is_delete=False,
        new_content=new,
        old_content=old,
    )
    assert flags.debian_patches_series_removal is True


def test_evaluate_flags_ignores_series_addition():
    old = "patch-a.patch\n"
    new = "patch-a.patch\npatch-b.patch\n"
    flags = guards.evaluate_flags(
        "debian/patches/series",
        is_delete=False,
        new_content=new,
        old_content=old,
    )
    assert flags.debian_patches_series_removal is False


def test_evaluate_flags_detects_control_version_change():
    old = "Build-Depends: foo (>= 1.0), bar\n"
    new = "Build-Depends: foo (>= 0.9), bar\n"
    flags = guards.evaluate_flags(
        "debian/control",
        is_delete=False,
        new_content=new,
        old_content=old,
    )
    assert flags.debian_control_version_change is True


def test_evaluate_flags_quiet_on_unrelated_changes():
    old = "Description: foo\n"
    new = "Description: foo bar\n"
    flags = guards.evaluate_flags(
        "debian/control",
        is_delete=False,
        new_content=new,
        old_content=old,
    )
    assert flags.debian_control_version_change is False
