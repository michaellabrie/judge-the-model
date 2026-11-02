#!/usr/bin/env bash
# Start the Streamlit app on this computer only. Stop it with Ctrl+C.
# It installs nothing, never runs git and publishes nothing (publishing is in SETUP.md).
#
#   bash run_app.sh

set -euo pipefail

# Run from this script's own folder, wherever it is called from.
cd "$(dirname "$0")"

# The project's Python environment, or the active python as a fallback.
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
  PY=".venv/Scripts/python.exe"
elif command -v python >/dev/null 2>&1; then
  PY="python"
  echo "Note: no .venv found here; using the active python ($(command -v python))."
else
  echo "Error: no Python found. Create the environment first (see SETUP.md, step 1):" >&2
  echo "  uv venv --python 3.11 && uv pip install -r requirements.txt" >&2
  exit 1
fi

if ! "$PY" -c "import streamlit" >/dev/null 2>&1; then
  echo "Error: Streamlit is not installed in $PY." >&2
  echo "Install the packages into your environment (see SETUP.md, step 1):" >&2
  echo "  uv pip install -r requirements.txt" >&2
  exit 1
fi

if [ ! -f "streamlit_app.py" ]; then
  echo "Error: streamlit_app.py is missing from $(pwd)." >&2
  exit 1
fi

PORT="${PORT:-8501}"
echo "Starting Judge the Model (Streamlit) on this computer only."
echo "Open http://localhost:${PORT} in your browser. Stop it with Ctrl+C."
exec "$PY" -m streamlit run streamlit_app.py --server.port "$PORT" --server.headless true --browser.gatherUsageStats false
