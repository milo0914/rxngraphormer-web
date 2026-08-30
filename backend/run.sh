#!/usr/bin/env bash
# run.sh — Local development runner for the backend
set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Activate venv
source /app/venv/bin/activate

# Set PYTHONPATH to include repo root (for rxngraphormer imports)
export PYTHONPATH="$REPO_ROOT"

# Change to repo root
cd "$REPO_ROOT"

# Run the FastAPI server
exec python -m backend.main