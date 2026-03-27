"""
routers/billing.py — Billing and API key management router for Summit.OS api-gateway.

Endpoints (all under /v1/billing):
  POST /v1/billing/keys               — generate a new API key for an org
  GET  /v1/billing/subscription       — return org tier, limits, entity count
  POST /v1/billing/checkout           — create a Stripe Checkout session
  POST /v1/billing/webhooks/stripe    — Stripe webhook handler (no SDK — stdlib hmac)
"""

from __future__ import annotations

import hashlib
import hmac
import json as _json_mod
import logging
import os
import secrets
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select, update

from middleware.billing import (
    OrgContext,
    TIER_DEFAULTS,
    _SessionLocal,
    _hash_key,
    api_keys_table,
    orgs_table,
    require_api_key,
    encrypt_field,
    decrypt_field,
)

logger = logging.getLogger("api-gateway.billing_router")

billing_router = APIRouter(prefix="/v1/billing", tags=["billing"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class CreateKeyRequest(BaseModel):
    org_id: str
    org_name: str
    tier: str = "free"


class CreateKeyResponse(BaseModel):
    api_key: str  # plaintext — shown ONCE, never stored
    org_id: str
    tier: str
    entity_limit: int
    operator_limit: int


class SubscriptionResponse(BaseModel):
    org_id: str
    tier: str
    subscription_status: str
    entity_limit: int
    operator_limit: int


class CheckoutRequest(BaseModel):
    org_id: str
    tier: str  # "pro" | "enterprise"
    success_url: str
    cancel_url: str


class CheckoutResponse(BaseModel):
    checkout_url: str
    session_id: str


# ---------------------------------------------------------------------------
# POST /v1/billing/keys
# ---------------------------------------------------------------------------


@billing_router.post("/keys", response_model=CreateKeyResponse, status_code=201)
async def create_api_key(body: CreateKeyRequest) -> CreateKeyResponse:
    """
    Generate a new API key for an org.

    Creates the org row if it does not already exist.
    The plaintext key is returned exactly once and never stored.
    """
    if _SessionLocal is None:
        raise HTTPException(status_code=503, detail="Billing service not initialised")

    tier = body.tier if body.tier in TIER_DEFAULTS else "free"
    defaults = TIER_DEFAULTS[tier]

    # Generate key: sk_live_<32 random hex chars>
    plaintext_key = "sk_live_" + secrets.token_hex(32)
    key_hash = _hash_key(plaintext_key)
    now = datetime.now(timezone.utc)

    async with _SessionLocal() as session:
        # Upsert org row — insert if not present
        existing_org = await session.execute(
            select(orgs_table).where(orgs_table.c.org_id == body.org_id)
        )
        if existing_org.first() is None:
            await session.execute(
                orgs_table.insert().values(
                    org_id=body.org_id,
                    name=encrypt_field(body.org_name),
                    tier=tier,
                    subscription_status="active",
                    entity_limit=defaults["entity_limit"],
                    operator_limit=defaults["operator_limit"],
                    created_at=now,
                )
            )

        # Insert the new key
        await session.execute(
            api_keys_table.insert().values(
                key_hash=key_hash,
                org_id=body.org_id,
                tier=tier,
                created_at=now,
                last_used_at=None,
                active=True,
            )
        )
        await session.commit()

    logger.info("Created API key for org=%s tier=%s", body.org_id, tier)

    return CreateKeyResponse(
        api_key=plaintext_key,
        org_id=body.org_id,
        tier=tier,
        entity_limit=defaults["entity_limit"],
        operator_limit=defaults["operator_limit"],
    )


# ---------------------------------------------------------------------------
# GET /v1/billing/subscription
# ---------------------------------------------------------------------------


@billing_router.get("/subscription", response_model=SubscriptionResponse)
async def get_subscription(
    ctx: OrgContext = Depends(require_api_key),
) -> SubscriptionResponse:
    """Return the org's current tier, limits, and subscription status."""
    if _SessionLocal is None:
        raise HTTPException(status_code=503, detail="Billing service not initialised")

    async with _SessionLocal() as session:
        org_result = await session.execute(
            select(orgs_table).where(orgs_table.c.org_id == ctx.org_id)
        )
        org_row = org_result.first()

    if org_row is None:
        # Key was valid but org row missing — surface free-tier info
        return SubscriptionResponse(
            org_id=ctx.org_id,
            tier=ctx.tier,
            subscription_status="active",
            entity_limit=ctx.entity_limit,
            operator_limit=ctx.operator_limit,
        )

    return SubscriptionResponse(
        org_id=ctx.org_id,
        tier=org_row.tier,
        subscription_status=org_row.subscription_status,
        entity_limit=org_row.entity_limit,
        operator_limit=org_row.operator_limit,
    )


# ---------------------------------------------------------------------------
# POST /v1/billing/checkout
# ---------------------------------------------------------------------------

_TIER_PRICE_ENV: dict[str, str] = {
    "pro": "STRIPE_PRICE_ID_PRO",
    "enterprise": "STRIPE_PRICE_ID_ENTERPRISE",
}


@billing_router.post("/checkout", response_model=CheckoutResponse, status_code=201)
async def create_checkout_session(body: CheckoutRequest) -> CheckoutResponse:
    """
    Create a Stripe Checkout session for the requested tier.

    Resolves STRIPE_SECRET_KEY and the price ID via the secrets client (Vault →
    env fallback).  No Stripe SDK — uses stdlib urllib.request, consistent with
    the webhook handler above.
    """
    try:
        import sys as _sys_ck
        from pathlib import Path as _Path_ck

        _ck_root = str(_Path_ck(__file__).resolve().parents[3])
        if _ck_root not in _sys_ck.path:
            _sys_ck.path.insert(0, _ck_root)
        from packages.secrets.client import get_secret as _get_ck

        stripe_key = (await _get_ck("STRIPE_SECRET_KEY", default="")) or ""
    except Exception:
        stripe_key = os.getenv("STRIPE_SECRET_KEY", "")

    if not stripe_key:
        raise HTTPException(status_code=503, detail="Stripe not configured")

    tier = body.tier.lower()
    if tier not in _TIER_PRICE_ENV:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown tier '{tier}'. Valid tiers: {list(_TIER_PRICE_ENV)}",
        )

    price_env_key = _TIER_PRICE_ENV[tier]
    try:
        from packages.secrets.client import get_secret as _get_price  # type: ignore[import]

        price_id = (await _get_price(price_env_key, default="")) or ""
    except Exception:
        price_id = os.getenv(price_env_key, "")

    if not price_id:
        raise HTTPException(
            status_code=503,
            detail=f"Price ID for tier '{tier}' not configured ({price_env_key})",
        )

    # Build Stripe Checkout session via urllib (no SDK)
    payload = urllib.parse.urlencode(
        {
            "mode": "subscription",
            "line_items[0][price]": price_id,
            "line_items[0][quantity]": "1",
            "client_reference_id": body.org_id,
            "success_url": body.success_url,
            "cancel_url": body.cancel_url,
        }
    ).encode()

    req = urllib.request.Request(
        "https://api.stripe.com/v1/checkout/sessions",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {stripe_key}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            session_data: dict = _json_mod.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        logger.error("Stripe checkout error %s: %s", exc.code, body_text)
        raise HTTPException(
            status_code=502,
            detail=f"Stripe returned {exc.code}: {body_text[:200]}",
        )
    except urllib.error.URLError as exc:
        logger.error("Stripe unreachable: %s", exc.reason)
        raise HTTPException(status_code=502, detail="Stripe service unreachable")

    checkout_url: str = session_data.get("url", "")
    session_id: str = session_data.get("id", "")

    if not checkout_url or not session_id:
        raise HTTPException(status_code=502, detail="Stripe response missing url or id")

    logger.info(
        "Created Stripe checkout session=%s org=%s tier=%s",
        session_id,
        body.org_id,
        tier,
    )
    return CheckoutResponse(checkout_url=checkout_url, session_id=session_id)


# ---------------------------------------------------------------------------
# POST /v1/billing/webhooks/stripe
# ---------------------------------------------------------------------------


def _verify_stripe_signature(payload: bytes, sig_header: str, secret: str) -> bool:
    """
    Verify a Stripe webhook signature using stdlib hmac (no stripe SDK).

    Stripe format: t=<timestamp>,v1=<hex_digest>[,v1=...]
    The signed payload is: "<timestamp>.<raw_body>"
    """
    try:
        parts: dict[str, list[str]] = {}
        for item in sig_header.split(","):
            if "=" not in item:
                continue
            k, v = item.split("=", 1)
            parts.setdefault(k.strip(), []).append(v.strip())

        timestamp = parts.get("t", [None])[0]
        v1_sigs = parts.get("v1", [])

        if not timestamp or not v1_sigs:
            return False

        signed_payload = f"{timestamp}.".encode() + payload
        expected = hmac.new(
            secret.encode(),
            signed_payload,
            hashlib.sha256,
        ).hexdigest()

        return any(hmac.compare_digest(expected, sig) for sig in v1_sigs)
    except Exception:
        return False


@billing_router.post("/webhooks/stripe")
async def stripe_webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(None, alias="stripe-signature"),
) -> dict:
    """
    Handle Stripe webhook events.

    Verifies the signature using STRIPE_WEBHOOK_SECRET env var.
    Handles:
      - customer.subscription.updated → update tier and status
      - customer.subscription.deleted → set status to cancelled
      - invoice.payment_failed       → set status to past_due
    """
    import json as _json

    try:
        import sys as _sys_sec
        from pathlib import Path as _Path_sec

        _sec_root = str(_Path_sec(__file__).resolve().parents[3])
        if _sec_root not in _sys_sec.path:
            _sys_sec.path.insert(0, _sec_root)
        from packages.secrets.client import get_secret as _get_sec

        webhook_secret = await _get_sec("STRIPE_WEBHOOK_SECRET", default="") or ""
    except Exception:
        webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    raw_body = await request.body()

    if webhook_secret:
        if not stripe_signature:
            raise HTTPException(
                status_code=400, detail="Missing stripe-signature header"
            )
        if not _verify_stripe_signature(raw_body, stripe_signature, webhook_secret):
            raise HTTPException(status_code=400, detail="Invalid Stripe signature")

    try:
        event = _json.loads(raw_body)
    except _json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    event_type: str = event.get("type", "")
    data_object: dict = event.get("data", {}).get("object", {})

    if _SessionLocal is None:
        logger.warning(
            "Billing session factory not ready — ignoring Stripe event %s", event_type
        )
        return {"received": True}

    async with _SessionLocal() as session:
        if event_type == "customer.subscription.updated":
            stripe_sub_id: str = data_object.get("id", "")
            new_status: str = data_object.get("status", "active")
            # Map Stripe status → internal status
            internal_status = (
                "cancelled"
                if new_status in ("canceled", "cancelled", "incomplete_expired")
                else "past_due" if new_status == "past_due" else "active"
            )
            # Determine tier from price metadata if available
            plan_name: str = (
                data_object.get("metadata", {}).get("tier")
                or data_object.get("plan", {}).get("nickname", "")
                or ""
            ).lower()
            tier = plan_name if plan_name in TIER_DEFAULTS else None

            values: dict = {"subscription_status": internal_status}
            if tier:
                defaults = TIER_DEFAULTS[tier]
                values["tier"] = tier
                values["entity_limit"] = defaults["entity_limit"]
                values["operator_limit"] = defaults["operator_limit"]

            if stripe_sub_id:
                await session.execute(
                    update(orgs_table)
                    .where(orgs_table.c.stripe_subscription_id == stripe_sub_id)
                    .values(**values)
                )
                await session.commit()
                logger.info(
                    "stripe event=%s sub=%s -> status=%s tier=%s",
                    event_type,
                    stripe_sub_id,
                    internal_status,
                    tier,
                )

        elif event_type == "customer.subscription.deleted":
            stripe_sub_id = data_object.get("id", "")
            if stripe_sub_id:
                await session.execute(
                    update(orgs_table)
                    .where(orgs_table.c.stripe_subscription_id == stripe_sub_id)
                    .values(subscription_status="cancelled")
                )
                await session.commit()
                logger.info(
                    "stripe event=%s sub=%s -> cancelled", event_type, stripe_sub_id
                )

        elif event_type == "invoice.payment_failed":
            stripe_customer_id: str = data_object.get("customer", "")
            if stripe_customer_id:
                await session.execute(
                    update(orgs_table)
                    .where(orgs_table.c.stripe_customer_id == stripe_customer_id)
                    .values(subscription_status="past_due")
                )
                await session.commit()
                logger.info(
                    "stripe event=%s customer=%s -> past_due",
                    event_type,
                    stripe_customer_id,
                )
        else:
            logger.debug("Unhandled Stripe event type: %s", event_type)

    return {"received": True}
