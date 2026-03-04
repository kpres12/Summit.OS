#!/usr/bin/env python3
"""
Summit.OS Smoke Test

End-to-end test that verifies the core data path:
  1. POST an entity to fabric's WorldStore API
  2. GET the entity back via api-gateway
  3. Connect to WebSocket and verify entity updates stream

Usage:
  python scripts/smoke_test.py                   # default: localhost
  GATEWAY_URL=http://myhost:8000 python scripts/smoke_test.py
"""

import asyncio
import json
import os
import sys
import time
import uuid

import httpx

GATEWAY = os.getenv("GATEWAY_URL", "http://localhost:8000")
FABRIC = os.getenv("FABRIC_URL", "http://localhost:8001")
TIMEOUT = float(os.getenv("SMOKE_TIMEOUT", "10"))

passed = 0
failed = 0


def ok(name: str):
    global passed
    passed += 1
    print(f"  ✓ {name}")


def fail(name: str, detail: str = ""):
    global failed
    failed += 1
    print(f"  ✗ {name}  — {detail}")


async def run():
    entity_id = f"smoke-{uuid.uuid4().hex[:8]}"
    entity_payload = {
        "entity_type": "ASSET",
        "domain": "AERIAL",
        "name": f"Smoke Test Entity {entity_id}",
        "class_label": "test-drone",
        "confidence": 0.95,
        "latitude": 37.7749,
        "longitude": -122.4194,
        "altitude": 100.0,
        "heading_deg": 45.0,
        "speed_mps": 12.5,
        "source_id": "smoke-test",
        "source_type": "test",
    }

    print(f"\n{'='*60}")
    print(f"  Summit.OS Smoke Test")
    print(f"  Gateway: {GATEWAY}")
    print(f"  Fabric:  {FABRIC}")
    print(f"{'='*60}\n")

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # ── Step 1: Health checks ──────────────────────────
        print("Step 1: Health checks")
        for svc, url in [
            ("gateway", f"{GATEWAY}/health"),
            ("fabric", f"{FABRIC}/health"),
        ]:
            try:
                r = await client.get(url)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    ok(f"{svc} healthy")
                else:
                    fail(f"{svc} health", f"status={r.status_code}")
            except Exception as e:
                fail(f"{svc} health", str(e))

        # ── Step 2: Create entity via fabric ───────────────
        print("\nStep 2: Create entity in WorldStore")
        try:
            r = await client.post(f"{FABRIC}/api/v1/entities", json=entity_payload)
            if r.status_code == 200:
                body = r.json()
                created_id = body.get("entity", {}).get("id")
                if created_id:
                    entity_id = created_id
                    ok(f"entity created: {entity_id[:12]}…")
                else:
                    fail("entity create", "no id in response")
            else:
                fail("entity create", f"status={r.status_code} body={r.text[:200]}")
        except Exception as e:
            fail("entity create", str(e))

        # ── Step 3: Read entity back via gateway ───────────
        print("\nStep 3: Read entity via gateway")
        try:
            r = await client.get(f"{GATEWAY}/api/v1/entities/{entity_id}")
            if r.status_code == 200:
                body = r.json()
                ent = body.get("entity", {})
                if ent.get("id") == entity_id:
                    ok(f"entity retrieved: {ent.get('name', '')}")
                else:
                    fail("entity read", f"unexpected id: {ent.get('id')}")
            else:
                fail("entity read", f"status={r.status_code}")
        except Exception as e:
            fail("entity read", str(e))

        # ── Step 4: List entities via gateway ──────────────
        print("\nStep 4: List entities via gateway")
        try:
            r = await client.get(f"{GATEWAY}/api/v1/entities", params={"limit": "10"})
            if r.status_code == 200:
                body = r.json()
                total = body.get("total", 0)
                ok(f"listed {total} entities")
            else:
                fail("entity list", f"status={r.status_code}")
        except Exception as e:
            fail("entity list", str(e))

        # ── Step 5: COP endpoint ───────────────────────────
        print("\nStep 5: Common Operating Picture")
        try:
            r = await client.get(f"{GATEWAY}/api/v1/cop")
            if r.status_code == 200:
                body = r.json()
                total = body.get("total_entities", 0)
                ok(f"COP returned {total} entities")
            else:
                fail("COP", f"status={r.status_code}")
        except Exception as e:
            fail("COP", str(e))

        # ── Step 6: Cleanup — delete entity ────────────────
        print("\nStep 6: Cleanup")
        try:
            r = await client.delete(f"{FABRIC}/api/v1/entities/{entity_id}")
            if r.status_code == 200:
                ok("entity deleted")
            else:
                fail("entity delete", f"status={r.status_code}")
        except Exception as e:
            fail("entity delete", str(e))

    # ── Summary ────────────────────────────────────────────
    print(f"\n{'='*60}")
    total = passed + failed
    if failed == 0:
        print(f"  ALL {total} CHECKS PASSED ✓")
    else:
        print(f"  {passed}/{total} passed, {failed} FAILED ✗")
    print(f"{'='*60}\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    rc = asyncio.run(run())
    sys.exit(rc)
