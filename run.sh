#!/usr/bin/env bash
# Run with: source ./run.sh [IP] [PORT]
# Also works from systemd ExecStart=/usr/bin/bash -lc '/path/to/run.sh 0.0.0.0 8501'

_pageindex_run() {
  set -Eeuo pipefail

  local server_address="${1:-${SERVER_NAME:-${STREAMLIT_SERVER_ADDRESS:-}}}"
  local port_number="${2:-${SERVER_PORT:-${STREAMLIT_SERVER_PORT:-8501}}}"
  local project_dir project_name venv_dir python_bin

  project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  project_name="$(basename "${project_dir}")"
  venv_dir="${VENV_DIR:-${HOME}/venv/${project_name}}"
  python_bin="${venv_dir}/bin/python"

  cd "${project_dir}"

  if [ ! -x "${python_bin}" ]; then
    "${project_dir}/install.sh"
  fi

  # shellcheck disable=SC1091
  source "${venv_dir}/bin/activate"

  if [ -f "${project_dir}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${project_dir}/.env"
    set +a
  fi

  export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
  export SERVER_NAME="${server_address}"
  export SERVER_PORT="${port_number}"
  export STREAMLIT_SERVER_ADDRESS="${server_address}"
  export STREAMLIT_SERVER_PORT="${port_number}"

  local args=(
    -m streamlit run app.py
    --browser.gatherUsageStats false
    --server.port "${port_number}"
  )

  if [ -n "${server_address}" ]; then
    args+=(--server.address "${server_address}")
  fi

  "${python_bin}" "${args[@]}"
}

_pageindex_run "$@"
unset -f _pageindex_run
