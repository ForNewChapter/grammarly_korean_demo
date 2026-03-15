#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TARGET_URL="http://${HOST}:${PORT}"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared is not installed. Install it first with: brew install cloudflared" >&2
  exit 1
fi

echo "[tunnel] forwarding ${TARGET_URL}"
echo "[tunnel] keep this terminal open"

exec cloudflared tunnel --url "$TARGET_URL"
