# ceph-autobuild-resolver

AI-driven build-failure resolution loop for the Ceph autobuild system. See
`../CE134 - Ceph Autobuild Loop.md` for the spec and
`../.claude/plan/CE134-architecture.md` for the architecture plan.

## Quickstart

```bash
cd resolver
pip install -e ".[test]"
pytest

# Run end-to-end (requires LXD + a prepared pristine container)
export OPENROUTER_API_KEY=sk-or-...
export MODEL_PROVIDER=openrouter
export MODEL_NAME=anthropic/claude-sonnet-4-5
ceph-autobuild-resolver --container ceph-build --pristine-snapshot pristine --matrix-name resolute
```

## Layout

```
src/ceph_autobuild_resolver/
  cli.py            Argparse entry point
  config.py         Env vars / Config dataclass
  orchestrator.py   Top-level: prep -> build -> [loop -> validate] -> output
  lxd.py            LXD container manager (lxc shell-out)
  build_runner.py   Wraps build.sh stages
  budget.py         Token, iteration, unchanged-iteration tracking
  loop.py           One round-trip with the model
  transcript.py     Append-only JSONL of every tool call & result
  prompts.py        System prompt + initial-context assembly
  guards.py         Edit-scope enforcement
  validation.py     Clean-rebuild driver
  providers/        Provider adapters (canonical schema <-> wire format)
  tools/            Tool schema + handlers + dispatcher
  output/           Launchpad PR / bug filing
```
