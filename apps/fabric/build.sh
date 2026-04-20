#!/usr/bin/env bash
set -e

# Install fabric dependencies
pip install -r requirements.txt

# Install shared package dependencies if present
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
for pkg in world entities geo sdk observability; do
  req="$REPO_ROOT/packages/$pkg/requirements.txt"
  if [ -f "$req" ]; then
    pip install -r "$req" || true
  fi
done
