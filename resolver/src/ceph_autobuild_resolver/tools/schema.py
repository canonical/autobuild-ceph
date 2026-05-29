"""Canonical declarations of every tool exposed to the model.

These declarations are provider-neutral — see ``providers/base.py``. The
adapter translates them into the provider's specific tool-declaration shape.

Every tool that returns paginated data also documents the metadata fields it
returns on truncation, so the model knows how to issue follow-up calls. See
the architecture plan's "Pagination & continuation" section.
"""

from __future__ import annotations

from ..providers.base import ToolSchema

DEFAULT_READ_FILE_LINES = 500
DEFAULT_GREP_MATCHES = 50
DEFAULT_LOG_BYTES = 65_536
DEFAULT_GIT_LOG_ENTRIES = 20


def all_tools() -> list[ToolSchema]:
    return [
        _read_files(),
        _read_log(),
        _grep_log(),
        _grep(),
        _git_log(),
        _replace_in_upstream(),
        _drop_patch(),
        _write_file(),
        _edit_file(),
        _apply_patch(),
        _delete_file(),
        _check_patch(),
        _run_build(),
        _clean(),
        _declare_resolved(),
        _declare_unresolvable(),
    ]


def _replace_in_upstream() -> ToolSchema:
    return ToolSchema(
        name="replace_in_upstream",
        description=(
            "PREFERRED tool for changing upstream Ceph source. Generates a "
            "properly-formatted debian/patches/<patch_name> that replaces "
            "old_content with new_content in the named file, with correct "
            "@@ headers (auto-computed) and DEP-3 metadata (auto-generated). "
            "Adds the entry to debian/patches/series. Use this instead of "
            "write_file on .patch files — you do not need to count lines or "
            "format diffs by hand. "
            "For multi-file patches, call this once per file with the SAME "
            "patch_name — each call appends a new diff block to the existing "
            "patch rather than replacing it."
        ),
        parameters={
            "type": "object",
            "properties": {
                "patch_name": {
                    "type": "string",
                    "description": (
                        "Filename for the new patch, e.g. 'fix-boost-resolver.patch'. "
                        "Must end with .patch."
                    ),
                },
                "file": {
                    "type": "string",
                    "description": (
                        "Path to the upstream file being changed, relative to "
                        "the working tree (e.g. 'src/common/Foo.cc')."
                    ),
                },
                "old_content": {
                    "type": "string",
                    "description": (
                        "Exact text to replace, copied verbatim from the file. "
                        "Must match exactly once — include a few surrounding "
                        "lines for uniqueness. Keep it SHORT: use a unique anchor, "
                        "never reproduce a whole file or large block. To wrap a "
                        "file (e.g. in a namespace), make two small calls anchored "
                        "on the first and closing lines — not one giant edit."
                    ),
                },
                "new_content": {
                    "type": "string",
                    "description": "Replacement text.",
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Short DEP-3 Description: line summarising what the "
                        "patch fixes. Becomes part of the PR."
                    ),
                },
            },
            "required": [
                "patch_name",
                "file",
                "old_content",
                "new_content",
                "description",
            ],
        },
    )


def _drop_patch() -> ToolSchema:
    return ToolSchema(
        name="drop_patch",
        description=(
            "Atomically remove a patch entry from debian/patches/series AND "
            "delete the .patch file. Use this when check_patch reports "
            "'Reversed (or previously applied) patch detected!' — that means "
            "the upstream source already contains those changes natively, "
            "and the right fix is removal, not rewriting."
        ),
        parameters={
            "type": "object",
            "properties": {
                "patch_name": {
                    "type": "string",
                    "description": "Filename without path, e.g. 'old-fix.patch'.",
                },
            },
            "required": ["patch_name"],
        },
    )


def _read_files() -> ToolSchema:
    return ToolSchema(
        name="read_files",
        description=(
            f"Read one or more files from the working tree. Each file is "
            f"capped at {DEFAULT_READ_FILE_LINES} lines unless start_line/"
            f"end_line are provided. Truncated responses include total_lines "
            f"so you can request the next slice."
        ),
        parameters={
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Paths relative to the ceph working tree.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "1-indexed inclusive start line. Default 1.",
                },
                "end_line": {
                    "type": "integer",
                    "description": (
                        "1-indexed inclusive end line. Default "
                        f"start_line + {DEFAULT_READ_FILE_LINES} - 1."
                    ),
                },
            },
            "required": ["paths"],
        },
    )


def _read_log() -> ToolSchema:
    return ToolSchema(
        name="read_log",
        description=(
            "Read a byte range from the captured build log. Useful when the "
            "tail returned in the initial context isn't enough to find the "
            "root cause. Response includes total_bytes for paginating."
        ),
        parameters={
            "type": "object",
            "properties": {
                "start": {"type": "integer", "description": "Byte offset, default 0."},
                "end": {
                    "type": "integer",
                    "description": (
                        "Exclusive byte offset, default "
                        f"start + {DEFAULT_LOG_BYTES}."
                    ),
                },
            },
        },
    )


def _grep_log() -> ToolSchema:
    return ToolSchema(
        name="grep_log",
        description=(
            "Search the captured build log for a pattern. Returns matching "
            "lines with surrounding context as raw text (format: "
            "'file:byteoffset:linenum:text'). PREFER this over read_log when "
            "looking for the root cause of a build failure — the log_tail "
            "in run_build's response only shows the first error excerpt, so "
            "for anything else (later errors, warnings, specific symbols) "
            "use grep_log. Case-insensitive by default."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Default true.",
                },
                "before_context": {
                    "type": "integer",
                    "description": "Lines of context before each match. Default 5.",
                },
                "after_context": {
                    "type": "integer",
                    "description": "Lines of context after each match. Default 20.",
                },
                "max_matches": {
                    "type": "integer",
                    "description": (
                        f"Cap returned matches; default {DEFAULT_GREP_MATCHES}."
                    ),
                },
            },
            "required": ["pattern"],
        },
    )


def _grep() -> ToolSchema:
    return ToolSchema(
        name="grep",
        description=(
            f"Search the working tree. Returns up to {DEFAULT_GREP_MATCHES} "
            "matches with surrounding excerpt; full files_with_matches list "
            "is always included so you can read_files directly even when "
            "match excerpts are truncated."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {
                    "type": "string",
                    "description": "Subtree to search; default is the repo root.",
                },
                "match_offset": {
                    "type": "integer",
                    "description": "Skip this many matches (for pagination).",
                },
                "max_matches": {
                    "type": "integer",
                    "description": (
                        f"Cap returned matches; default {DEFAULT_GREP_MATCHES}."
                    ),
                },
                "case_insensitive": {"type": "boolean"},
            },
            "required": ["pattern"],
        },
    )


def _git_log() -> ToolSchema:
    return ToolSchema(
        name="git_log",
        description="Show commit history; optionally filter by path.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "n": {
                    "type": "integer",
                    "description": f"Number of commits, default {DEFAULT_GIT_LOG_ENTRIES}.",
                },
                "offset": {"type": "integer"},
            },
        },
    )


def _write_file() -> ToolSchema:
    return ToolSchema(
        name="write_file",
        description=(
            "Replace the contents of a file. REJECTED for paths outside "
            "debian/ — express upstream changes as quilt patches under "
            "debian/patches/."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    )


def _edit_file() -> ToolSchema:
    return ToolSchema(
        name="edit_file",
        description=(
            "Replace exactly one occurrence of old_str with new_str. Fails "
            "if old_str is missing or appears more than once. Same scope "
            "rules as write_file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
            },
            "required": ["path", "old_str", "new_str"],
        },
    )


def _apply_patch() -> ToolSchema:
    return ToolSchema(
        name="apply_patch",
        description=(
            "Apply a unified diff. Each hunk's target path is scope-checked "
            "individually."
        ),
        parameters={
            "type": "object",
            "properties": {
                "diff": {
                    "type": "string",
                    "description": "Unified diff text.",
                },
            },
            "required": ["diff"],
        },
    )


def _delete_file() -> ToolSchema:
    return ToolSchema(
        name="delete_file",
        description="Delete a file. Same scope rules as write_file.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )


def _check_patch() -> ToolSchema:
    return ToolSchema(
        name="check_patch",
        description=(
            "Dry-run a patch file with 'patch -F 0 -p1 --dry-run' — the same "
            "flags dpkg-source uses. Call this after replace_in_upstream creates "
            "or updates a debian/patches/*.patch file to verify hunk headers and "
            "context lines are correct before calling run_build. Returns ok=true "
            "only when all hunks apply cleanly."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Path to the patch file relative to the ceph working "
                        "tree, e.g. 'debian/patches/fix-foo.patch'."
                    ),
                },
            },
            "required": ["path"],
        },
    )


def _run_build() -> ToolSchema:
    return ToolSchema(
        name="run_build",
        description=(
            "Re-run the build stage in the iteration container. Returns the "
            "tail of the captured log and an ok flag. Use after applying "
            "fixes to test whether the failure is resolved."
        ),
        parameters={"type": "object", "properties": {}},
    )


def _clean() -> ToolSchema:
    return ToolSchema(
        name="clean",
        description=(
            "Remove transient build artefacts in the working tree (does NOT "
            "reset the container). Use sparingly; the iteration container "
            "preserves cache by design."
        ),
        parameters={"type": "object", "properties": {}},
    )


def _declare_unresolvable() -> ToolSchema:
    return ToolSchema(
        name="declare_unresolvable",
        description=(
            "Terminal call. Use ONLY when you have exhausted all reasonable "
            "approaches and are confident the build failure cannot be fixed "
            "with the tools and information available. Provide a clear account "
            "of what you tried, why each approach failed, and what additional "
            "information or changes would be needed. The harness will record "
            "your explanation, file a bug, and exit."
        ),
        parameters={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": (
                        "Explanation of what was tried, why it failed, and "
                        "what would be needed to resolve it."
                    ),
                },
            },
            "required": ["reason"],
        },
    )


def _declare_resolved() -> ToolSchema:
    return ToolSchema(
        name="declare_resolved",
        description=(
            "Terminal call. Invoke ONLY when run_build succeeded and you "
            "believe the proposed changes constitute the fix. The harness "
            "will then validate by clean-rebuilding from a pristine "
            "snapshot with only your diff applied."
        ),
        parameters={
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "Human-readable summary of the fix; goes into the PR."
                    ),
                },
            },
            "required": ["summary"],
        },
    )
