# ceph-autobuild-resolver

AI-driven build-failure resolution loop for the Ceph autobuild system.

When a Ceph Debian package build fails in CI, this tool feeds the failure to an
LLM (Gemini or OpenRouter/Claude), which uses a constrained set of tools
(file read/edit, grep, run_build, patch management) to diagnose and fix the
failure. If the model succeeds, the resulting diff is validated against a clean
container before being submitted as a PR.

## Quickstart

```bash
# Install
cd resolver
pip install -e .

# 1. Prepare a pristine container (one-time, or when dependencies change)
export UBUNTU_BRANCH=ubuntu/resolute
export DEBIAN_REF=origin/ubuntu/latest
export CEPH_VERSION=20.2.0
ceph-autobuild-resolver prep \
    --image ubuntu:24.04 \
    --container ceph-build \
    --snapshot pristine

# 2. Run the resolver against a failing build
export MODEL_PROVIDER=gemini                   # or openrouter
export GEMINI_API_KEY=AIza...                  # or OPENROUTER_API_KEY=sk-or-...
export MODEL_NAME=gemini-3.1-pro-preview       # or anthropic/claude-sonnet-4-5
ceph-autobuild-resolver resolve \
    --container ceph-build \
    --pristine-snapshot pristine \
    --matrix-name resolute
```

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_PROVIDER` | `gemini` | `gemini` or `openrouter` |
| `MODEL_NAME` | `gemini-3.1-pro-preview` / `anthropic/claude-sonnet-4-5` | Provider-specific model identifier (default varies by provider) |
| `GEMINI_API_KEY` | — | Required when `MODEL_PROVIDER=gemini` |
| `OPENROUTER_API_KEY` | — | Required when `MODEL_PROVIDER=openrouter` |
| `MAX_ITERATIONS` | `20` | Maximum model turns before giving up |
| `MAX_UNCHANGED_ITERATIONS` | `3` | Consecutive turns with no file changes before aborting (anti-spin) |
| `RUN_TOKEN_BUDGET` | `32000000` | Total token budget; 0 = unlimited |
| `MAX_WALL_SECONDS` | `0` (disabled) | Wall-clock time limit in seconds |
| `MAX_SECONDS_TO_FIRST_BUILD` | `0` (disabled) | Abort if no run_build call within this many seconds |
| `UBUNTU_BRANCH` | `ubuntu/resolute` | Ubuntu packaging branch to check out |
| `DEBIAN_REF` | `origin/ubuntu/latest` | Debian packaging ref |
| `LAUNCHPAD_OWNER` | `lmlogiudice` | Launchpad username for packaging repo |
| `CEPH_VERSION` | `20.2.0` | Upstream Ceph release to build |
| `CCACHE_HOST_DIR` | — | Host path for a ccache bind-mount (optional speedup) |
| `REASONING_EFFORT` | — | OpenRouter only: `low`, `medium`, or `high` thinking effort |
| `REASONING_MAX_TOKENS` | — | OpenRouter only: explicit thinking token budget (overrides `REASONING_EFFORT`) |

## How the loop works

```
prep   → launch container from image → install deps → clone repo + tarball
         → install build deps → snapshot (pristine)

resolve → restore container to pristine snapshot
        → initial build attempt   (if OK: done)
        → deterministic preflight (drop reversed patches, refresh drifted ones)
        → re-build                (if OK: validate + PR)
        → AI resolution loop:
              model sees: system prompt + file tree + build failure + patch series
              model uses: read_files / grep / edit_file / replace_in_upstream /
                          run_build / check_patch / declare_resolved / ...
              loop exits: declare_resolved → validate → PR
                          declare_unresolvable → file bug
                          budget exhausted → file bug
        → validation: copy pristine snapshot → apply only model's diff → build
        → output: CI diff + narrative (dry_run) or Launchpad PR (production)
```

The edit scope is enforced: the model may only modify files under `debian/`.
Changes to upstream source must be expressed as quilt patches under
`debian/patches/` using the `replace_in_upstream` tool.

## Adding a new LLM provider

1. Create `src/ceph_autobuild_resolver/providers/<name>.py`.
2. Implement the `ProviderAdapter` protocol from `providers/base.py`:
   - `declare_tools(tools)` — translate `ToolSchema` list to the provider's format.
   - `chat(history)` — translate the canonical `Message` list, call the API,
     return `(Message, Usage)` with arguments already deserialized to dicts.
3. Register it in `providers/factory.py`.

## Adding a new tool

1. Add a `_tool_name()` function in `tools/schema.py` and include it in `all_tools()`.
2. Add a handler method to the appropriate handler class in `tools/`.
3. Wire it into `Dispatcher._handlers` in `tools/dispatch.py`.
4. If the tool mutates files, add the name to `_FILE_MUTATORS` in `tools/dispatch.py`.

## Layout

```
src/ceph_autobuild_resolver/
  cli.py            Argparse entry point (prep / resolve subcommands)
  config.py         Env vars → frozen Config dataclass
  orchestrator.py   Top-level: restore → build → [loop → validate] → output
  lxd.py            LXD container manager (pylxd wrapper)
  build_runner.py   Executes build stages inside the container
  build_steps.py    Pure-Python stage / step definitions
  prep.py           Pristine container preparation
  budget.py         Token, iteration, and no-progress tracking
  loop.py           Resolution loop: model ↔ tool dispatch
  transcript.py     Append-only JSONL of every model turn and tool result
  prompts.py        System prompt + initial-context assembly
  guards.py         Edit-scope enforcement (debian/ only)
  validation.py     Clean-rebuild driver (pristine snapshot + model diff)
  display.py        Rich terminal UI helpers
  providers/        LLM provider adapters (canonical ↔ wire format)
  tools/            Tool schema declarations, dispatcher, and handlers
  output/           CI output, Launchpad PR creation, and bug filing
```
