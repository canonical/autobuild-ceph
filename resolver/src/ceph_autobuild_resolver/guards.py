"""Edit-scope enforcement.

Per the spec: edits to upstream Ceph source files (anything outside ``debian/``)
must be expressed as new or modified entries under ``debian/patches/``. The
guard rejects direct mutations to upstream source paths so that all
behavioural changes are reviewable as discrete patches and stay compatible
with the existing quilt-based packaging workflow.

A second, softer guard *flags* (does not block) modifications to
``debian/control`` version constraints and removals from
``debian/patches/series`` — these go into the PR summary for explicit
reviewer attention.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class EditScopeViolation(Exception):
    """Raised when a write/edit/delete targets a path outside the allowed
    scope. The dispatcher converts this into a structured tool error so the
    model can correct course on its next turn."""


@dataclass(frozen=True)
class WriteFlags:
    """Soft warnings raised by the second-tier guard. None of these block the
    write; they're surfaced in the PR summary."""

    debian_control_version_change: bool = False
    debian_patches_series_removal: bool = False
    system_flag_disabled: bool = False  # WITH_SYSTEM_* set to OFF in a patch


def normalize(path: str) -> str:
    """Strip leading ``./`` and collapse repeated separators."""
    while path.startswith("./"):
        path = path[2:]
    while "//" in path:
        path = path.replace("//", "/")
    return path.lstrip("/")


def contained_relpath(path: str, workdir: str | None = None) -> str:
    """Normalize a read-side path and refuse anything outside the workspace.

    Unlike :func:`assert_in_scope` (which restricts *writes* to ``debian/``),
    read tools may touch the whole working tree — but never beyond it: build
    logs are PR-controlled input to the model, so traversal out of the tree
    (e.g. onto the host ccache bind-mount via ``../ccache``) must be treated
    as adversarial. ``..`` segments are always rejected, and an absolute path
    is accepted only when it points inside ``workdir``; the workdir prefix is
    stripped so callers always get a workdir-relative path back.
    """
    is_abs = path.startswith("/")
    p = normalize(path)
    if ".." in p.split("/"):
        raise EditScopeViolation(
            f"path {path!r} contains '..' — reads are restricted to the "
            "build workspace"
        )
    if workdir:
        workdir_rel = normalize(workdir)
        if p == workdir_rel:
            return ""
        if p.startswith(workdir_rel + "/"):
            return p[len(workdir_rel) + 1:]
        if is_abs:
            raise EditScopeViolation(
                f"absolute path {path!r} is outside the workspace "
                f"{workdir!r} — pass a path relative to the working tree"
            )
    return p


# Script extensions that must never be written into the working tree.
# The model occasionally tries to create helper programs to work around
# tool limitations; this guard catches those attempts early.
_BLOCKED_EXTENSIONS = frozenset({".py", ".sh", ".rb", ".pl", ".php"})


def assert_in_scope(path: str) -> None:
    """Raise ``EditScopeViolation`` if ``path`` is outside ``debian/`` or is
    a script file."""
    p = normalize(path)
    # ``..`` traversal is always rejected; we don't try to canonicalise across
    # symlinks here because the path is logical (relative to the working dir
    # in-container), not a host filesystem path.
    if ".." in p.split("/"):
        raise EditScopeViolation(
            f"path {path!r} contains '..' — refused for safety"
        )
    filename = p.rsplit("/", 1)[-1]
    if "." in filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower()
        if ext in _BLOCKED_EXTENSIONS:
            raise EditScopeViolation(
                f"path {path!r}: writing {ext} script files into the tree is "
                "not allowed. Use the available tools directly — "
                "replace_in_upstream for upstream changes, edit_file / "
                "write_file for debian/ packaging files."
            )
    if p == "debian" or p.startswith("debian/"):
        return
    raise EditScopeViolation(
        f"path {path!r} is outside debian/ and cannot be edited directly. "
        "Use replace_in_upstream to make upstream source changes — it "
        "generates the correct unified diff and updates debian/patches/ "
        "automatically."
    )


def assert_not_patch_file(path: str) -> None:
    """Raise ``EditScopeViolation`` if ``path`` targets a .patch file under debian/patches/.

    Patch files MUST be managed through replace_in_upstream (to create/append) and
    drop_patch (to remove). Direct edits produce incorrect @@ line numbers and will
    fail at dpkg-source time.  This guard is applied to edit_file, write_file, and
    delete_file — but NOT to replace_in_upstream or drop_patch, which are the
    authoritative patch managers.
    """
    p = normalize(path)
    if p.startswith("debian/patches/") and p.endswith(".patch"):
        name = p.rsplit("/", 1)[-1]
        raise EditScopeViolation(
            f"Direct edit of patch file {name!r} is not allowed. "
            "Patch files must only be managed by replace_in_upstream (to add/update hunks) "
            "or drop_patch (to remove the patch entirely). "
            "Direct edits produce incorrect @@ line numbers and break the build."
        )


def evaluate_flags(
    path: str,
    *,
    is_delete: bool,
    new_content: str | None,
    old_content: str | None,
) -> WriteFlags:
    """Inspect a write/delete to surface reviewer-attention flags.

    Cheap heuristics; perfect detection is not the goal — the PR is human
    reviewed regardless.
    """
    p = normalize(path)
    flags = WriteFlags()

    if p == "debian/patches/series":
        # A line removal here is suspicious — it disables an existing patch.
        if old_content is not None and new_content is not None:
            old_lines = {
                line.strip()
                for line in old_content.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            }
            new_lines = {
                line.strip()
                for line in new_content.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            }
            if old_lines - new_lines:
                flags = _with(flags, debian_patches_series_removal=True)
        elif is_delete:
            flags = _with(flags, debian_patches_series_removal=True)

    if p == "debian/control" and old_content is not None and new_content is not None:
        if _control_version_constraints_changed(old_content, new_content):
            flags = _with(flags, debian_control_version_change=True)

    if p.startswith("debian/patches/") and new_content is not None:
        if _system_flag_disabled_in_patch(new_content):
            flags = _with(flags, system_flag_disabled=True)

    return flags


def _with(flags: WriteFlags, **kwargs: bool) -> WriteFlags:
    return WriteFlags(
        debian_control_version_change=kwargs.get(
            "debian_control_version_change", flags.debian_control_version_change
        ),
        debian_patches_series_removal=kwargs.get(
            "debian_patches_series_removal", flags.debian_patches_series_removal
        ),
        system_flag_disabled=kwargs.get(
            "system_flag_disabled", flags.system_flag_disabled
        ),
    )


def _control_version_constraints_changed(old: str, new: str) -> bool:
    """Heuristic: did any line containing a version constraint change?

    We extract lines that look like dependency constraints — anything
    containing ``(>=``, ``(<<``, ``(=``, ``(>>``, ``(<=`` — and compare sets.
    """
    markers = ("(>=", "(<<", "(=", "(>>")

    def constraint_lines(text: str) -> set[str]:
        return {
            line.strip()
            for line in text.splitlines()
            if any(m in line for m in markers)
        }

    return constraint_lines(old) != constraint_lines(new)


def _system_flag_disabled_in_patch(patch_content: str) -> bool:
    """Return True if the patch adds a line that sets any WITH_SYSTEM_* to OFF.

    Scans only added lines (starting with '+' but not '+++') so we don't
    trigger on context lines or removed lines.
    """
    # Matches cmake set() or option() calls that turn a WITH_SYSTEM_* off.
    _off_pattern = re.compile(
        r"WITH_SYSTEM_\w+\s+(OFF|False|0)\b", re.IGNORECASE
    )
    for line in patch_content.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            if _off_pattern.search(line):
                return True
    return False
