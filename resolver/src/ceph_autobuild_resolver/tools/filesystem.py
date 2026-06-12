"""Filesystem-side tool handlers (read, write, edit, patch, delete).

All paths are interpreted relative to the working tree inside the LXD
container. Handlers route through the LXD manager so the tools work the same
whether we're hitting a real container in CI or a fake in unit tests.
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass
from typing import Any

from .. import display, guards
from ..build_runner import BuildRunner
from ..config import TOOL_EXEC_TIMEOUT_SECONDS, Config
from ..lxd import LXDManager

log = logging.getLogger(__name__)

# Whole-file reads (edit_file/write_file/replace_in_upstream internals) are
# capped at the source via head -c. Comfortably above the dispatcher's
# 256 KiB per-result clamp and far above any legitimate packaging file.
_MAX_FULL_READ_BYTES = 512 * 1024


@dataclass
class FilesystemHandlers:
    lxd: LXDManager
    cfg: Config
    container: str
    runner: BuildRunner

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def read_files(
        self,
        paths: list[str],
        start_line: int = 1,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        from .schema import DEFAULT_READ_FILE_LINES

        files: list[dict[str, Any]] = []
        for raw in paths:
            # contained_relpath also strips a workdir prefix from absolute
            # paths (e.g. "/root/ceph/debian/control") so we don't
            # double-prepend below. A traversal or out-of-workspace path is
            # a per-file error, not a whole-call failure.
            try:
                rel = guards.contained_relpath(raw, self.cfg.container_workdir)
            except guards.EditScopeViolation as exc:
                files.append({"path": raw, "error": str(exc)})
                continue
            full = f"{self.cfg.container_workdir}/{rel}"
            count_res = self.lxd.exec(
                self.container,
                ["wc", "-l", full],
                check=False,
                timeout=TOOL_EXEC_TIMEOUT_SECONDS,
            )
            if not count_res.ok:
                files.append(
                    {"path": rel, "error": count_res.stderr.strip() or "read failed"}
                )
                continue
            try:
                total_lines = int(count_res.stdout.strip().split()[0])
            except (ValueError, IndexError):
                total_lines = 0

            if total_lines > 0 and start_line > total_lines:
                files.append({
                    "path": rel,
                    "content": "",
                    "start_line": start_line,
                    "end_line": total_lines,
                    "total_lines": total_lines,
                    "truncated": False,
                    "note": (
                        f"start_line ({start_line}) is beyond end of file "
                        f"({total_lines} lines) — use a smaller start_line"
                    ),
                })
                continue

            effective_end = end_line or (start_line + DEFAULT_READ_FILE_LINES - 1)
            # ``sed -n 'a,bp'`` is portable and 1-indexed inclusive on both ends.
            slice_res = self.lxd.exec(
                self.container,
                ["sed", "-n", f"{start_line},{effective_end}p", full],
                check=False,
                timeout=TOOL_EXEC_TIMEOUT_SECONDS,
            )
            content = slice_res.stdout
            files.append(
                {
                    "path": rel,
                    "content": content,
                    "start_line": start_line,
                    "end_line": min(effective_end, total_lines or effective_end),
                    "total_lines": total_lines,
                    "truncated": effective_end < total_lines,
                }
            )
        return {"files": files}

    def grep_log(
        self,
        log_path: str,
        pattern: str,
        case_insensitive: bool = True,
        before_context: int = 5,
        after_context: int = 20,
        max_matches: int | None = None,
    ) -> dict[str, Any]:
        """Search the captured build log by pattern.

        Returns each match with surrounding context, byte offset (so the
        model can pivot to ``read_log`` for a wider window), and a count
        of total matches. Defaults to case-insensitive because most build
        errors capitalise inconsistently (``error:``, ``ERROR:``, ``Error``).
        """
        from .schema import (
            DEFAULT_GREP_MATCHES,
            HARD_MAX_CONTEXT_LINES,
            HARD_MAX_MATCHES,
        )

        # Clamp every input the model controls: an explicit large value
        # (e.g. after_context=5000, max_matches=10000) would make grep buffer a
        # huge result into stdout before the exec timeout / 256 KB payload clamp
        # can fire. Mirrors the caps the workspace grep already applies.
        max_matches = max_matches or DEFAULT_GREP_MATCHES
        max_matches = max(1, min(max_matches, HARD_MAX_MATCHES))
        before_context = max(0, min(before_context, HARD_MAX_CONTEXT_LINES))
        after_context = max(0, min(after_context, HARD_MAX_CONTEXT_LINES))
        # ``-b`` prints byte offsets so the model can read_log around any
        # match. ``-A``/``-B`` give surrounding context. ``-m`` caps grep's
        # own work — we still slice in Python in case multiple match groups
        # appeared on the same line.
        # --text forces grep to treat the file as text even if it contains
        # ANSI escape codes or other byte sequences cmake sometimes writes.
        # Without this, grep may silently return no matches on cmake logs.
        flags = ["-nbH", "--text", f"-A{after_context}", f"-B{before_context}"]
        if case_insensitive:
            flags.append("-i")
        flags.append(f"-m{max_matches}")
        result = self.lxd.exec(
            self.container,
            ["grep", *flags, "-e", pattern, log_path],
            check=False,
            timeout=TOOL_EXEC_TIMEOUT_SECONDS,
        )
        # grep returns 1 when there are no matches, which is not an error.
        if result.returncode > 1:
            return {
                "error": result.stderr.strip() or "grep_log failed",
                "pattern": pattern,
            }
        return {
            "pattern": pattern,
            "log_path": log_path,
            "output": result.stdout,
            "any_matches": bool(result.stdout.strip()),
        }

    def read_log(
        self,
        log_path: str,
        start: int = 0,
        end: int | None = None,
    ) -> dict[str, Any]:
        from .schema import DEFAULT_LOG_BYTES

        end = end if end is not None else start + DEFAULT_LOG_BYTES
        size_res = self.lxd.exec(
            self.container,
            ["stat", "-c", "%s", log_path],
            check=False,
            timeout=TOOL_EXEC_TIMEOUT_SECONDS,
        )
        if not size_res.ok:
            return {"error": size_res.stderr.strip() or "log not found"}
        total_bytes = int(size_res.stdout.strip() or 0)
        # ``dd`` with bs=1 is slow on large files; using a count derived from
        # the requested window keeps the slice cheap.
        count = max(0, min(end, total_bytes) - start)
        if count == 0:
            return {
                "content": "",
                "start": start,
                "end": start,
                "total_bytes": total_bytes,
                "truncated": False,
            }
        result = self.lxd.exec(
            self.container,
            [
                "dd",
                f"if={log_path}",
                "bs=1",
                f"skip={start}",
                f"count={count}",
                "status=none",
            ],
            check=False,
            timeout=TOOL_EXEC_TIMEOUT_SECONDS,
        )
        return {
            "content": result.stdout,
            "start": start,
            "end": start + count,
            "total_bytes": total_bytes,
            "truncated": (start + count) < total_bytes,
        }

    # ------------------------------------------------------------------
    # Writes — all go through guards.assert_in_scope first
    # ------------------------------------------------------------------

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        guards.assert_in_scope(path)
        guards.assert_not_patch_file(path)
        rel = guards.normalize(path)
        full = f"{self.cfg.container_workdir}/{rel}"

        old = self._read_full_if_exists(full)
        flags = guards.evaluate_flags(
            rel, is_delete=False, new_content=content, old_content=old
        )
        self.lxd.put_text(self.container, full, content)
        display.file_written(rel, old, content)
        return {"ok": True, "flags": _flags_to_dict(flags)}

    def edit_file(
        self, path: str, old_str: str, new_str: str
    ) -> dict[str, Any]:
        guards.assert_in_scope(path)
        guards.assert_not_patch_file(path)
        rel = guards.normalize(path)
        full = f"{self.cfg.container_workdir}/{rel}"

        old = self._read_full_if_exists(full)
        if old is None:
            return {"ok": False, "error": f"file not found: {rel}"}
        occurrences = old.count(old_str)
        if occurrences == 0:
            return {"ok": False, "error": "old_str not found"}
        if occurrences > 1:
            return {
                "ok": False,
                "error": f"old_str matches {occurrences} times — make it unique",
            }
        new_content = old.replace(old_str, new_str, 1)
        flags = guards.evaluate_flags(
            rel, is_delete=False, new_content=new_content, old_content=old
        )
        self.lxd.put_text(self.container, full, new_content)
        display.file_edited(rel, old, new_content)
        return {"ok": True, "flags": _flags_to_dict(flags)}

    def apply_patch(self, diff: str) -> dict[str, Any]:
        # Scope-check every path git apply will actually touch before applying.
        # The authoritative path list comes from `git apply --numstat` (git's
        # own resolver, post -p1), not from parsing the diff text: a text
        # parser is easy to slip past (an `x/`-prefixed, header-only, or
        # mode-only diff yields no targets yet git apply still writes the
        # file). numstat reports create/modify/delete/rename-to targets;
        # rename/copy *sources* (which numstat omits) come from the literal
        # `rename from`/`copy from` headers, which are repo-relative.
        # Refuse symlink creation/conversion: a `mode 120000` entry creates a
        # symlink under debian/ that passes the logical-path scope check, and a
        # later read/write through it dereferences outside the workspace (the
        # guards deliberately don't resolve symlinks). Path scoping alone can't
        # see this, so reject before applying.
        if _diff_creates_symlink(diff):
            raise guards.EditScopeViolation(
                "patch creates or converts a file to a symlink (mode 120000); "
                "refused — symlinks can escape the workspace when dereferenced."
            )
        numstat = self.runner.patch_numstat(self.container, diff)
        if not numstat.ok:
            return {
                "ok": False,
                "stderr": (
                    "git apply could not parse this diff (--numstat failed); "
                    "the patch is malformed and was not applied. "
                    + (numstat.stderr.strip() or "")
                ).strip(),
            }
        targets = set(_numstat_paths(numstat.stdout)) | set(_rename_copy_sources(diff))
        # The patch-file block (assert_not_patch_file) must also be enforced
        # here: apply_patch can otherwise bypass the hard block that write_file
        # and edit_file enforce, allowing the model to directly write malformed
        # @@ headers into debian/patches/*.patch files.
        for path in sorted(targets):
            guards.assert_in_scope(path)
            guards.assert_not_patch_file(path)
        # apply_patch_to_tree does NOT reset the working tree, so all prior
        # model edits (patches written, rules changed) are preserved.
        result = self.runner.apply_patch_to_tree(self.container, diff)
        return {
            "ok": result.ok,
            "stderr": result.stderr if not result.ok else "",
        }

    def delete_file(self, path: str) -> dict[str, Any]:
        guards.assert_in_scope(path)
        guards.assert_not_patch_file(path)
        rel = guards.normalize(path)
        full = f"{self.cfg.container_workdir}/{rel}"
        old = self._read_full_if_exists(full)
        flags = guards.evaluate_flags(
            rel, is_delete=True, new_content=None, old_content=old
        )
        result = self.lxd.exec(
            self.container,
            ["rm", "-f", full],
            check=False,
            timeout=TOOL_EXEC_TIMEOUT_SECONDS,
        )
        return {
            "ok": result.ok,
            "flags": _flags_to_dict(flags),
            "stderr": result.stderr if not result.ok else "",
        }

    # ------------------------------------------------------------------
    # Structural patch authoring (preferred over raw write_file on .patch)
    # ------------------------------------------------------------------

    def replace_in_upstream(
        self,
        patch_name: str,
        file: str,
        old_content: str,
        new_content: str,
        description: str,
    ) -> dict[str, Any]:
        """Generate a debian/patches/<patch_name> for an upstream change.

        Reads the current upstream file, validates ``old_content`` matches
        exactly once, generates a unified diff with correct ``@@`` headers via
        difflib, wraps it with DEP-3 headers, writes the patch and (if
        needed) appends an entry to debian/patches/series.
        """
        if not patch_name or patch_name.startswith("#") or "/" in patch_name:
            return {"ok": False, "error": "invalid patch_name"}
        if not patch_name.endswith(".patch"):
            return {"ok": False, "error": "patch_name must end with .patch"}
        if old_content == new_content:
            return {"ok": False, "error": "old_content equals new_content — no change"}

        # Contain the read like the other read tools (read_files/grep/git_log):
        # guards.normalize alone keeps '..', so an adversarial file like
        # '../../etc/shadow' or '../ccache/...' would be read and its contents
        # leaked into the generated patch under debian/patches/ (then readable
        # via read_files). contained_relpath rejects traversal/out-of-workspace.
        try:
            rel_file = guards.contained_relpath(file, self.cfg.container_workdir)
        except guards.EditScopeViolation as exc:
            return {"ok": False, "error": str(exc)}
        full_file = f"{self.cfg.container_workdir}/{rel_file}"
        current = self._read_full_if_exists(full_file)
        if current is None:
            return {"ok": False, "error": f"upstream file not found: {rel_file}"}

        occ = current.count(old_content)
        if occ == 0:
            # Help the model understand where the mismatch starts so it can
            # correct old_content without a full re-read.
            hint = ""
            if old_content:
                # Find the longest prefix of old_content that IS in the file.
                for length in range(min(len(old_content), 120), 0, -1):
                    if old_content[:length] in current:
                        snippet = repr(old_content[:length])
                        hint = (
                            f" First {length} chars matched but diverged after that "
                            f"(matched prefix: {snippet})."
                        )
                        break
                if not hint:
                    snippet = repr(old_content[:60])
                    hint = f" Even the first characters were not found (started with: {snippet})."
            return {
                "ok": False,
                "error": (
                    f"old_content not found in {rel_file}.{hint} "
                    "Tip: use a short anchor (10-20 lines) rather than reproducing "
                    "the entire section — exact whitespace must match the file."
                ),
            }
        if occ > 1:
            return {
                "ok": False,
                "error": (
                    f"old_content matches {occ} times in {rel_file} — include "
                    "more surrounding context so it is unique."
                ),
            }

        new_file = current.replace(old_content, new_content, 1)
        diff_lines = list(
            difflib.unified_diff(
                current.splitlines(keepends=True),
                new_file.splitlines(keepends=True),
                fromfile=f"a/{rel_file}",
                tofile=f"b/{rel_file}",
                n=3,
            )
        )

        diff_text = "".join(diff_lines)
        if not diff_text.endswith("\n"):
            diff_text += "\n"

        # Write the patch, appending to any existing one so that multi-file
        # patches accumulate correctly across successive calls.
        rel_patch = f"debian/patches/{patch_name}"
        guards.assert_in_scope(rel_patch)
        full_patch = f"{self.cfg.container_workdir}/{rel_patch}"
        old_patch = self._read_full_if_exists(full_patch)

        if old_patch:
            # Guard against appending a second hunk for the same file: each
            # hunk is computed against the unmodified on-disk file, so two
            # independent hunks for the same file would have overlapping/
            # contradictory context and dpkg-source would reject the patch.
            already_patched = f"a/{rel_file}" in old_patch
            if already_patched:
                return {
                    "ok": False,
                    "error": (
                        f"{rel_file} already has a hunk in {patch_name}. "
                        "To make multiple changes to the same file, combine "
                        "them into one call: widen old_content to cover all "
                        "the lines you want to replace."
                    ),
                }
            # Append diff block; DEP-3 headers from the first call stay.
            patch_content = old_patch.rstrip("\n") + "\n" + diff_text
        else:
            headers = (
                f"Description: {description}\n"
                f"Author: ceph-autobuild-resolver\n"
                f"Forwarded: not-needed\n"
                f"\n"
            )
            patch_content = headers + diff_text

        flags = guards.evaluate_flags(
            rel_patch, is_delete=False, new_content=patch_content, old_content=old_patch
        )
        self.lxd.put_text(self.container, full_patch, patch_content)
        display.file_written(rel_patch, old_patch, patch_content)

        # Append to series if not already there.
        series_rel = "debian/patches/series"
        full_series = f"{self.cfg.container_workdir}/{series_rel}"
        series = self._read_full_if_exists(full_series) or ""
        existing = {
            line.strip()
            for line in series.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        appended = False
        if patch_name not in existing:
            new_series = series.rstrip("\n")
            if new_series:
                new_series += "\n"
            new_series += patch_name + "\n"
            self.lxd.put_text(self.container, full_series, new_series)
            appended = True

        return {
            "ok": True,
            "patch_path": rel_patch,
            "appended_to_series": appended,
            "flags": _flags_to_dict(flags),
        }

    def drop_patch(self, patch_name: str) -> dict[str, Any]:
        """Atomically remove ``patch_name`` from series and delete the file."""
        if not patch_name or "/" in patch_name:
            return {"ok": False, "error": "invalid patch_name"}

        rel_patch = f"debian/patches/{patch_name}"
        guards.assert_in_scope(rel_patch)
        full_patch = f"{self.cfg.container_workdir}/{rel_patch}"

        # Update series: drop any line whose stripped form equals patch_name.
        series_rel = "debian/patches/series"
        full_series = f"{self.cfg.container_workdir}/{series_rel}"
        series = self._read_full_if_exists(full_series) or ""
        kept: list[str] = []
        removed_from_series = False
        for line in series.splitlines():
            if line.strip() == patch_name:
                removed_from_series = True
                continue
            kept.append(line)
        new_series = "\n".join(kept)
        if new_series and not new_series.endswith("\n"):
            new_series += "\n"
        if removed_from_series:
            self.lxd.put_text(self.container, full_series, new_series)

        # Delete the patch file (no error if it never existed).
        old_patch = self._read_full_if_exists(full_patch)
        existed = old_patch is not None
        if existed:
            self.lxd.exec(
                self.container,
                ["rm", "-f", full_patch],
                check=False,
                timeout=TOOL_EXEC_TIMEOUT_SECONDS,
            )
            display.file_written(rel_patch, old_patch, "")

        return {
            "ok": True,
            "removed_from_series": removed_from_series,
            "deleted_file": existed,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read_full_if_exists(self, full_path: str) -> str | None:
        # Cap the read at the source: a huge file (e.g. a build artifact)
        # must not land fully in host memory just for the dispatcher to
        # discard it. Refusing (rather than truncating) matters: edit_file
        # writes the full content back, so a silent truncation would corrupt
        # the file.
        result = self.lxd.exec(
            self.container,
            ["head", "-c", str(_MAX_FULL_READ_BYTES + 1), full_path],
            check=False,
            timeout=TOOL_EXEC_TIMEOUT_SECONDS,
        )
        if not result.ok:
            return None
        if len(result.stdout.encode("utf-8", "replace")) > _MAX_FULL_READ_BYTES:
            raise ValueError(
                f"{full_path} exceeds {_MAX_FULL_READ_BYTES} bytes -- "
                "refusing a whole-file read. Files this large are build "
                "artifacts, not editable packaging files."
            )
        return result.stdout


def _flags_to_dict(flags: guards.WriteFlags) -> dict[str, bool]:
    return {
        "debian_control_version_change": flags.debian_control_version_change,
        "debian_patches_series_removal": flags.debian_patches_series_removal,
        "system_flag_disabled": flags.system_flag_disabled,
    }


def _numstat_paths(stdout: str) -> list[str]:
    """Parse ``git apply --numstat -z`` output into the paths it touches.

    Records are NUL-delimited. A normal record is ``<added>\\t<deleted>\\t<path>``;
    some git versions emit a rename as ``<added>\\t<deleted>\\t`` followed by the
    old and new paths as separate NUL-terminated tokens. Counts are always tab-
    joined with their path, so a standalone (tab-less) token is a rename/copy
    path, never a numeric count.
    """
    paths: list[str] = []
    for tok in stdout.split("\0"):
        if not tok:
            continue
        if "\t" in tok:
            tail = tok.split("\t")[-1]
            if tail:
                paths.append(tail)
        else:
            paths.append(tok)
    return paths


def _diff_creates_symlink(diff: str) -> bool:
    """True if a unified diff creates a symlink or converts a file to one.

    git encodes symlinks as mode 120000. We flag any ``mode 120000`` on a
    create/change line; deleting a symlink (also 120000) is harmless but rare
    enough that refusing the whole class via apply_patch is acceptable — the
    model has delete_file/drop_patch for removals.
    """
    for line in diff.splitlines():
        s = line.strip()
        if s.endswith("120000") and (
            s.startswith("new file mode")
            or s.startswith("new mode")
            or s.startswith("old mode")
            or s.startswith("deleted file mode")
        ):
            return True
    return False


def _rename_copy_sources(diff: str) -> list[str]:
    """Repo-relative source paths from ``rename from``/``copy from`` headers.

    numstat reports only the rename/copy *destination*; the source (which the
    operation deletes/reads) must be scope-checked too. These header lines are
    literal repo-relative paths, independent of the diff's ``a/``/``b/`` prefix.
    """
    sources: list[str] = []
    for line in diff.splitlines():
        for prefix in ("rename from ", "copy from "):
            if line.startswith(prefix):
                p = line[len(prefix):].strip()
                if p:
                    sources.append(p)
    return sources
