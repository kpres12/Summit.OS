#!/bin/bash
set -e

FABRIC_PORT=${PORT:-10000}
NEXTJS_PORT=3001

echo "Starting Heli.OS — fabric on :$FABRIC_PORT, frontend on :$NEXTJS_PORT"

# Start Next.js standalone server on internal port
cd /app/apps/console-server
PORT=$NEXTJS_PORT HOSTNAME=127.0.0.1 node server.js &
NEXTJS_PID=$!

# Give Next.js a moment to start
sleep 2

# Start Python fabric — it proxies unmatched routes to Next.js
export NEXTJS_INTERNAL_URL="http://127.0.0.1:$NEXTJS_PORT"
cd /app/apps/fabric
exec uvicorn main:app --host 0.0.0.0 --port $FABRIC_PORT --workers 1
