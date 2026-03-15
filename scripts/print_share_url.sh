#!/usr/bin/env bash
set -euo pipefail

SITE_URL="${1:-https://fornewchapter.github.io/grammarly_korean_demo/}"
API_BASE="${2:-}"

if [[ -z "$API_BASE" ]]; then
  echo "usage: $0 <site-url> <api-base-url>" >&2
  echo "example: $0 https://fornewchapter.github.io/grammarly_korean_demo/ https://abc.trycloudflare.com" >&2
  exit 1
fi

python3 - "$SITE_URL" "$API_BASE" <<'PY'
from urllib.parse import urlencode
import sys

site_url = sys.argv[1]
api_base = sys.argv[2]
sep = '&' if '?' in site_url else '?'
print(f"{site_url}{sep}{urlencode({'apiBase': api_base})}")
PY
