#!/usr/bin/env bash
# scripts/setup-polar.sh
#
# One-time Polar.sh product setup for Summit.OS.
# Creates Pro, Organization, and Enterprise products and returns
# the product IDs to paste into your .env / Coolify environment.
#
# Requirements:
#   export POLAR_ACCESS_TOKEN=polar_at_...
#   (Settings → API → Create token with "products:write" scope)
#
# Run:
#   bash scripts/setup-polar.sh
#
# Output:
#   POLAR_PRODUCT_ID_PRO=...
#   POLAR_PRODUCT_ID_ORG=...
#   POLAR_PRODUCT_ID_ENTERPRISE=...

set -euo pipefail

if [[ -z "${POLAR_ACCESS_TOKEN:-}" ]]; then
  echo "ERROR: POLAR_ACCESS_TOKEN is not set." >&2
  echo "  → polar.sh → Settings → API → New token (products:write)" >&2
  echo "  export POLAR_ACCESS_TOKEN=polar_at_..." >&2
  exit 1
fi

API="https://api.polar.sh/v1"
AUTH="Authorization: Bearer ${POLAR_ACCESS_TOKEN}"

polar_post() {
  local endpoint="$1"
  local data="$2"
  curl -sS -X POST "$API/$endpoint" \
    -H "$AUTH" \
    -H "Content-Type: application/json" \
    -d "$data"
}

polar_id() {
  python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('id','ERROR: '+str(d)))"
}

echo "Creating Summit.OS products on Polar.sh..."
echo ""

# ── Pro ───────────────────────────────────────────────────────────────────────

PRO_ID=$(polar_post "products" '{
  "name": "Summit.OS Pro",
  "description": "500 entities · 5 operators. COMMAND view, mission planning, sensor fusion, and webhook integrations.",
  "prices": [
    {
      "type": "recurring",
      "recurring_interval": "month",
      "price_amount": 4900,
      "price_currency": "usd"
    }
  ],
  "metadata": { "tier": "pro" }
}' | polar_id)

echo "Pro product: $PRO_ID"

# ── Organization ──────────────────────────────────────────────────────────────

ORG_ID=$(polar_post "products" '{
  "name": "Summit.OS Organization",
  "description": "5,000 entities · unlimited operators. DEV view, adapter registry, multi-user RBAC, and audit log.",
  "prices": [
    {
      "type": "recurring",
      "recurring_interval": "month",
      "price_amount": 19900,
      "price_currency": "usd"
    }
  ],
  "metadata": { "tier": "org" }
}' | polar_id)

echo "Organization product: $ORG_ID"

# ── Enterprise ────────────────────────────────────────────────────────────────

ENT_ID=$(polar_post "products" '{
  "name": "Summit.OS Enterprise",
  "description": "Unlimited entities and operators. Multi-tenant, SSO/SAML, on-prem or air-gapped. Custom SLA.",
  "prices": [
    {
      "type": "recurring",
      "recurring_interval": "year",
      "price_amount": 0,
      "price_currency": "usd"
    }
  ],
  "metadata": { "tier": "enterprise" }
}' | polar_id)

echo "Enterprise product: $ENT_ID (set price to $0 — sales will quote custom)"

# ── Webhook ───────────────────────────────────────────────────────────────────

echo ""
echo "Register your Polar webhook:"
echo "  polar.sh → Settings → Webhooks → Add endpoint"
echo "  URL: https://YOUR_DOMAIN/v1/billing/webhooks/polar"
echo "  Events:"
echo "    subscription.created"
echo "    subscription.updated"
echo "    subscription.revoked"
echo ""
echo "Copy the webhook secret → POLAR_WEBHOOK_SECRET"

# ── Output ────────────────────────────────────────────────────────────────────

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Add these to your .env / Coolify environment variables:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "POLAR_PRODUCT_ID_PRO=${PRO_ID}"
echo "POLAR_PRODUCT_ID_ORG=${ORG_ID}"
echo "POLAR_PRODUCT_ID_ENTERPRISE=${ENT_ID}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
