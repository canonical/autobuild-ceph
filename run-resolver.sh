#!/usr/bin/env bash
set -euo pipefail

# uv run requires pyproject.toml in (or above) the CWD.
cd "$(dirname "$0")/resolver"

# Container name. CI sets $CONTAINER to a per-run-unique name
# (ceph-build-<matrix>-<run_id>) so concurrent jobs on one LXD host never
# share a container; fall back to a per-matrix name for local runs.
MATRIX_NAME="${MATRIX_NAME:-resolute}"
CONTAINER="${CONTAINER:-ceph-build-${MATRIX_NAME}}"

exec uv run ceph-autobuild-resolver resolve \
  --container "${CONTAINER}" \
  --pristine-snapshot pristine \
  --matrix-name "${MATRIX_NAME}" \
  --dry-run-output \
  "$@"
