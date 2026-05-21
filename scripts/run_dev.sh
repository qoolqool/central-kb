#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

CENTRAL_KB_DB_PATH="${CENTRAL_KB_DB_PATH:-/data/central-kb.sqlite3}"

echo "Starting Central KB..."
echo "  DB: $CENTRAL_KB_DB_PATH"

exec uvicorn app.main:app --host 0.0.0.0 --port 9000 --reload
