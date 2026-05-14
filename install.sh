#!/usr/bin/env bash
# Idempotent installer/upgrader for pageindex-mvp.
# Creates/updates ~/venv/<basename project dir> using uv.

set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "${PROJECT_DIR}")"
VENV_DIR="${VENV_DIR:-${HOME}/venv/${PROJECT_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
UV_BIN="${UV_BIN:-uv}"

log() {
  printf '[install] %s\n' "$*"
}

ensure_uv() {
  if command -v "${UV_BIN}" >/dev/null 2>&1; then
    return 0
  fi

  log "uv not found; installing with ${PYTHON_BIN} -m pip --user"
  "${PYTHON_BIN}" -m pip install --user --upgrade uv

  if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
    export PATH="${HOME}/.local/bin:${PATH}"
  fi

  command -v "${UV_BIN}" >/dev/null 2>&1
}

main() {
  cd "${PROJECT_DIR}"
  ensure_uv

  mkdir -p "$(dirname "${VENV_DIR}")"
  log "creating/updating virtualenv: ${VENV_DIR}"
  "${UV_BIN}" venv --python "${PYTHON_BIN}" "${VENV_DIR}"

  log "installing/upgrading dependencies"
  "${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" --upgrade pip setuptools wheel
  "${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" --upgrade -r "${PROJECT_DIR}/requirements.txt"

  if [ ! -f "${PROJECT_DIR}/.env" ] && [ -f "${PROJECT_DIR}/.env.example" ]; then
    cp "${PROJECT_DIR}/.env.example" "${PROJECT_DIR}/.env"
    log "created .env from .env.example"
  fi

  log "done"
  log "activate: source '${VENV_DIR}/bin/activate'"
  log "run:      source '${PROJECT_DIR}/run.sh' [IP] [PORT]"
}

main "$@"
