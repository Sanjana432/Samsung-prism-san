#!/usr/bin/env bash
# Checked cross-platform Python runner; every failed stage stops the workflow.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
exec "$PY" scripts/run_pipeline.py "$@"
