#!/usr/bin/env bash
set -euo pipefail

# uv run requires pyproject.toml in (or above) the CWD.
cd "$(dirname "$0")/resolver"

exec uv run ceph-autobuild-resolver resolve \
  --container ceph-build \
  --pristine-snapshot pristine \
  --matrix-name "${MATRIX_NAME:-resolute}" \
  --dry-run-output \
  "$@"
