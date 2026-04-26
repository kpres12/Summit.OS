#!/usr/bin/env bash
# TAK Server interop test — run from a workstation with Docker available.
#
# Brings up the TAK Server compose stack (taky reference image), runs the
# pytest integration test against it, then tears down. Use this to
# validate atak_adapter against a real CoT producer before any
# AFRL Rome / CANVAS demo.
#
# Prerequisites:
#   - Docker Desktop running
#   - python venv with pytest, pytest-asyncio installed
#
# Exit codes:
#   0  all tests passed
#   1  TAK Server failed to come up
#   2  test failures
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/infra/docker/docker-compose.tak.yml"

cleanup() {
  echo "[tak-interop] tearing down..."
  docker compose -f "$COMPOSE_FILE" down -v --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[tak-interop] bringing up TAK Server..."
docker compose -f "$COMPOSE_FILE" up -d tak-server

echo "[tak-interop] waiting for TAK Server to accept CoT on :8087..."
for i in $(seq 1 60); do
  if docker compose -f "$COMPOSE_FILE" exec -T tak-server sh -c "nc -z localhost 8087" >/dev/null 2>&1; then
    echo "[tak-interop] TAK Server up after ${i}s"
    break
  fi
  sleep 1
  if [ "$i" = "60" ]; then
    echo "[tak-interop] TAK Server failed to start after 60s" >&2
    docker compose -f "$COMPOSE_FILE" logs tak-server | tail -50 >&2
    exit 1
  fi
done

echo "[tak-interop] running integration tests..."
ATAK_SERVER_HOST=localhost ATAK_SERVER_PORT=8087 \
  "$REPO_ROOT/.venv/bin/python3.13" -m pytest \
  "$REPO_ROOT/tests/integration/test_atak_interop.py" -v --tb=short
