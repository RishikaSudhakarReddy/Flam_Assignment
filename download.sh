#!/usr/bin/env bash
# download.sh - Downloads raw Indic corpora for the pipeline
set -euo pipefail

RAW_DIR="${1:-data/raw}"
echo "Downloading raw datasets into: ${RAW_DIR}"

if command -v python3 &> /dev/null; then
    python3 download.py "${RAW_DIR}"
elif command -v python &> /dev/null; then
    python download.py "${RAW_DIR}"
else
    echo "Error: Python is required to execute download.py" >&2
    exit 1
fi

echo "Download completed successfully."
