#!/usr/bin/env bash
# run.sh - Offline Indic corpus cleaning and curation pipeline
# Usage: ./run.sh <raw_dir> <out_dir>
set -euo pipefail

RAW_DIR="${1:-data/raw}"
OUT_DIR="${2:-data/out}"

if [ ! -d "$RAW_DIR" ]; then
    echo "Error: raw input directory '${RAW_DIR}' does not exist." >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

if command -v python3 &> /dev/null; then
    python3 src/pipeline.py "${RAW_DIR}" "${OUT_DIR}"
elif command -v python &> /dev/null; then
    python src/pipeline.py "${RAW_DIR}" "${OUT_DIR}"
else
    echo "Error: Python 3 runtime could not be located in PATH." >&2
    exit 1
fi
