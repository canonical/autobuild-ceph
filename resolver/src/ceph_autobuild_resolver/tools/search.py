"""grep and git_log handlers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..config import Config
from ..lxd import LXDManager

log = logging.getLogger(__name__)


@dataclass
class SearchHandlers:
    lxd: LXDManager
    cfg: Config
    container: str

    def grep(
        self,
        pattern: str,
        path: str = ".",
        match_offset: int = 0,
        max_matches: int | None = None,
        case_insensitive: bool = False,
    ) -> dict[str, Any]:
        from .schema import (
            DEFAULT_GREP_MATCHES,
            HARD_MAX_MATCHES,
            MAX_EXCERPT_CHARS,
            MAX_FILES_LISTED,
        )

        # Distinguish "omitted" (None -> default) from an explicit integer, so
        # max_matches=0 means "just counts/files, no excerpts" rather than
        # being coerced to the default. Then clamp into [0, HARD_MAX_MATCHES]:
        # the ceiling stops a huge value reintroducing an unbounded payload,
        # the floor stops a negative value turning the slice below into a
        # nonsensical matches[offset:offset-n].
        if max_matches is None:
            max_matches = DEFAULT_GREP_MATCHES
        max_matches = max(0, min(max_matches, HARD_MAX_MATCHES))

        # Normalise path: if the model passes an absolute container path
        # (e.g. "/root/ceph/CMakeLists.txt") strip the workdir prefix so we
        # don't double-prepend and end up with a non-existent path.
        if path.startswith("/"):
            workdir = self.cfg.container_workdir.rstrip("/")
            if path.startswith(workdir + "/"):
                path = path[len(workdir) + 1:]
            elif path == workdir:
                path = "."
            # else: absolute path outside workdir — pass as-is and let grep fail
        # We rely on grep's recursive mode and let the shell's path resolution
        # apply relative to the workdir. ``-n`` prefixes line numbers; ``-H``
        # ensures path is always shown (even on a single-file target).
        flags = ["-rnH", "--binary-files=without-match"]
        if case_insensitive:
            flags.append("-i")
        argv = [
            "grep",
            *flags,
            "-e",
            pattern,
            f"{self.cfg.container_workdir}/{path}",
        ]
        # grep returns 1 when there are no matches, which is not an error.
        result = self.lxd.exec(self.container, argv, check=False)
        all_lines = result.stdout.splitlines()
        # Sometimes grep emits paths as absolute (because we pass an absolute
        # search root). Strip the workdir prefix so the model sees repo-rel.
        prefix = self.cfg.container_workdir.rstrip("/") + "/"

        matches: list[dict[str, Any]] = []
        files_with_matches: set[str] = set()
        for raw in all_lines:
            # Format: ``<path>:<line>:<excerpt>``. Path may itself contain
            # colons on weird filesystems but those don't appear in the Ceph
            # tree, so a 2-split is safe.
            parts = raw.split(":", 2)
            if len(parts) < 3:
                continue
            p, lineno_str, excerpt = parts
            if p.startswith(prefix):
                p = p[len(prefix):]
            files_with_matches.add(p)
            try:
                lineno = int(lineno_str)
            except ValueError:
                continue
            # Truncate the excerpt so a single huge (e.g. minified) line can't
            # bloat an otherwise-capped page.
            if len(excerpt) > MAX_EXCERPT_CHARS:
                excerpt = excerpt[:MAX_EXCERPT_CHARS] + " ...[excerpt truncated]"
            matches.append({"path": p, "line": lineno, "excerpt": excerpt})

        total = len(matches)
        page = matches[match_offset : match_offset + max_matches]

        # Bound files_with_matches: a broad pattern can match tens of thousands
        # of files, and returning the full list overflows the provider's
        # per-request token limit. Cap the list, report the true total, and
        # signal truncation so the model narrows its pattern rather than paging.
        all_files = sorted(files_with_matches)
        files_truncated = len(all_files) > MAX_FILES_LISTED
        result: dict[str, Any] = {
            "pattern": pattern,
            "matches": page,
            "match_offset": match_offset,
            "matches_returned": len(page),
            "total_matches": total,
            # Only meaningful when excerpts were actually requested: with
            # max_matches=0 the empty page is intentional, not a truncation.
            "truncated": max_matches > 0 and match_offset + len(page) < total,
            "files_with_matches": all_files[:MAX_FILES_LISTED],
            "total_files": len(all_files),
            "files_truncated": files_truncated,
        }
        if files_truncated:
            result["hint"] = (
                f"{len(all_files)} files matched but only the first "
                f"{MAX_FILES_LISTED} are listed. Narrow your pattern or pass a "
                "more specific path instead of paging the whole list."
            )
        return result

    def git_log(
        self,
        path: str | None = None,
        n: int | None = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        from .schema import DEFAULT_GIT_LOG_ENTRIES

        n = n or DEFAULT_GIT_LOG_ENTRIES
        argv = [
            "git",
            "-C",
            self.cfg.container_workdir,
            "log",
            f"--max-count={n}",
            f"--skip={offset}",
            "--pretty=format:%H%x09%an%x09%ad%x09%s",
            "--date=iso",
        ]
        if path:
            argv.extend(["--", path])
        result = self.lxd.exec(self.container, argv, check=False)
        if not result.ok:
            return {"error": result.stderr.strip() or "git log failed"}
        commits = []
        for raw in result.stdout.splitlines():
            parts = raw.split("\t", 3)
            if len(parts) != 4:
                continue
            sha, author, date, subject = parts
            commits.append(
                {"sha": sha, "author": author, "date": date, "subject": subject}
            )
        return {"commits": commits, "offset": offset, "returned": len(commits)}
