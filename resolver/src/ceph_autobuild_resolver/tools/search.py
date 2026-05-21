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
        from .schema import DEFAULT_GREP_MATCHES

        max_matches = max_matches or DEFAULT_GREP_MATCHES

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
            matches.append({"path": p, "line": lineno, "excerpt": excerpt})

        total = len(matches)
        page = matches[match_offset : match_offset + max_matches]
        return {
            "pattern": pattern,
            "matches": page,
            "match_offset": match_offset,
            "matches_returned": len(page),
            "total_matches": total,
            "files_with_matches": sorted(files_with_matches),
            "truncated": match_offset + len(page) < total,
        }

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
