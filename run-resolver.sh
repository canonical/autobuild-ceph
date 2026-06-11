#!/usr/bin/env bash
set -euo pipefail

# uv run requires pyproject.toml in (or above) the CWD.
cd "$(dirname "$0")/resolver"

# Derive the container name from the matrix so two matrix jobs (or queued
# runs from different PRs) on the same LXD host can never prep/restore/build
# in each other's container.
MATRIX_NAME="${MATRIX_NAME:-resolute}"

exec uv run ceph-autobuild-resolver resolve \
  --container "ceph-build-${MATRIX_NAME}" \
  --pristine-snapshot pristine \
  --matrix-name "${MATRIX_NAME}" \
  --dry-run-output \
  "$@"
