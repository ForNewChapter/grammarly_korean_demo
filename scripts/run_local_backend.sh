#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
DEFAULT_BACKEND_NAME="$(scutil --get ComputerName 2>/dev/null || hostname)"
DEFAULT_BACKEND_BRANCH="$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
DEFAULT_BACKEND_VERSION="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo no-git)"

if [[ -n "$(git -C "$ROOT_DIR" status --porcelain 2>/dev/null)" ]]; then
  DEFAULT_BACKEND_VERSION="${DEFAULT_BACKEND_VERSION}-dirty"
fi

export BACKEND_NAME="${BACKEND_NAME:-$DEFAULT_BACKEND_NAME}"
export BACKEND_BRANCH="${BACKEND_BRANCH:-$DEFAULT_BACKEND_BRANCH}"
export BACKEND_VERSION="${BACKEND_VERSION:-$DEFAULT_BACKEND_VERSION}"

cd "$ROOT_DIR"

echo "[backend] root=$ROOT_DIR"
echo "[backend] listening on http://${HOST}:${PORT}"
echo "[backend] identity=${BACKEND_NAME} / ${BACKEND_BRANCH} / ${BACKEND_VERSION}"

exec python3 -m uvicorn server:app --app-dir "$ROOT_DIR/apps/backend" --host "$HOST" --port "$PORT"
