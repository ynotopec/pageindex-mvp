#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-0.0.0.0}"
PORT="${2:-8501}"

cd "$(dirname "$0")"

export PYTHONUNBUFFERED=1

echo "Starting pageindex-mvp"
echo "Host: $HOST"
echo "Port: $PORT"
echo

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not installed or not in PATH"
  exit 1
fi

echo "Python used by uv:"
uv run python - <<'PY'
import sys
print(sys.executable)
print(sys.version)
PY

echo
echo "Checking PageIndex import:"
uv run python - <<'PY'
try:
    import pageindex
    print("pageindex file:", pageindex.__file__)
    from pageindex import PageIndexClient
    print("PageIndexClient OK:", PageIndexClient)
except Exception:
    import traceback
    traceback.print_exc()
    raise SystemExit(1)
PY

echo
echo "Starting Streamlit..."
exec uv run streamlit run app.py \
  --server.address "$HOST" \
  --server.port "$PORT"
