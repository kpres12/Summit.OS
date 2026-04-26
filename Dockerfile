# ── Stage 1: Build Next.js frontend ──────────────────────────────────────────
FROM node:20-alpine AS frontend-builder
WORKDIR /console

COPY apps/console/package.json apps/console/yarn.lock ./
RUN yarn install --frozen-lockfile

COPY apps/console/ ./

# Baked at build time — empty string = same-origin API calls
ENV NEXT_PUBLIC_API_URL=""
ENV NEXT_PUBLIC_WS_URL=""
ENV NEXT_PUBLIC_DEV_BYPASS_AUTH="true"
ENV NEXT_PUBLIC_APP_URL=""

RUN yarn build

# ── Stage 2: Python backend + Node runtime ────────────────────────────────────
FROM python:3.11-slim

# Install Node.js 20
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY apps/fabric/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Fabric service + shared packages (preserving relative paths for imports)
COPY apps/fabric/ ./apps/fabric/
COPY packages/ ./packages/

# Next.js standalone server
COPY --from=frontend-builder /console/.next/standalone ./apps/console-server/
COPY --from=frontend-builder /console/.next/static ./apps/console-server/.next/static/
COPY --from=frontend-builder /console/public ./apps/console-server/public/

COPY docker-start.sh ./
RUN chmod +x docker-start.sh


# ── Non-root runtime user (for K8s runAsNonRoot enforcement) ──
RUN groupadd --system --gid 65534 heli || true \
 && useradd  --system --uid 65534 --gid 65534 --no-create-home heli || true \
 && chown -R heli:heli /app
USER heli:heli
CMD ["./docker-start.sh"]
