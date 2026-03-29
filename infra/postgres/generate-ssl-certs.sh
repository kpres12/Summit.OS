#!/usr/bin/env bash
# Generate a self-signed TLS certificate for the Postgres container.
# Run once before first `docker compose up`. The certs are gitignored.
#
# Usage: bash infra/postgres/generate-ssl-certs.sh

set -euo pipefail

DEST="$(dirname "$0")/ssl"
mkdir -p "${DEST}"

echo "Generating self-signed Postgres TLS certificate..."

openssl req -new -x509 \
  -days 3650 \
  -nodes \
  -newkey rsa:4096 \
  -keyout "${DEST}/server.key" \
  -out    "${DEST}/server.crt" \
  -subj   "/CN=summit-postgres/O=Summit.OS/C=US" \
  -addext "subjectAltName=DNS:postgres,DNS:localhost,IP:127.0.0.1"

# Postgres requires the key to be owned by the postgres user (uid 999)
# and not readable by others.
chmod 600 "${DEST}/server.key"
chmod 644 "${DEST}/server.crt"

echo "Certs written to ${DEST}/"
echo ""
echo "NOTE: The postgres Docker container will chown these to uid 999 on first start."
echo "      If you see SSL permission errors, run:"
echo "        docker compose exec postgres chown postgres:postgres /etc/postgresql/ssl/server.key"
