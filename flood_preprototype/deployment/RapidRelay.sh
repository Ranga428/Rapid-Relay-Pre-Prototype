#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# activate the venv (Linux path)
source "$SCRIPT_DIR/../../floodenv/bin/activate"

python Start.py --schedule