#!/usr/bin/env bash
set -euo pipefail

exec uv run ceph-autobuild-resolver resolve \
  --container ceph-build \
  --pristine-snapshot pristine \
  --matrix-name resolute \
  --dry-run-output \
  "$@"
