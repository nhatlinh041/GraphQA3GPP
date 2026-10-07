#!/usr/bin/env bash
# Generate cloudflared-demo.yml from the template + .env (do NOT commit the output).
set -e

# Go to the repo root
cd "$(dirname "$0")/.."

# Load .env
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
else
  echo "✗ .env not found at $(pwd)/.env" >&2
  exit 1
fi

# Required variables
: "${CLOUDFLARED_TUNNEL_ID:?CLOUDFLARED_TUNNEL_ID is not set in .env}"
: "${CLOUDFLARED_CREDENTIALS_FILE:?CLOUDFLARED_CREDENTIALS_FILE is not set in .env}"
: "${CLOUDFLARED_HOSTNAME:?CLOUDFLARED_HOSTNAME is not set in .env}"

TEMPLATE="cloudflared-demo.template.yml"
OUTPUT="cloudflared-demo.yml"

if [ ! -f "$TEMPLATE" ]; then
  echo "✗ Template not found: $TEMPLATE" >&2
  exit 1
fi

# Substitute exactly the 3 whitelisted variables; leave any other ${} in the YAML alone
envsubst '${CLOUDFLARED_TUNNEL_ID} ${CLOUDFLARED_CREDENTIALS_FILE} ${CLOUDFLARED_HOSTNAME}' \
  < "$TEMPLATE" > "$OUTPUT"

echo "✓ Wrote $OUTPUT"
