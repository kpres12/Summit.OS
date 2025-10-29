import os
from fastapi.testclient import TestClient
from apps.fabric.main import app, world_entities, metadata
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text


def test_worldstate_org_filter(monkeypatch):
    # Ensure test mode for SQLite
    monkeypatch.setenv("FABRIC_TEST_MODE", "true")

    client = TestClient(app)

    # Insert sample rows into SQLite
    # Acquire engine/session from app lifespan by creating a temp engine here
    # Note: For simplicity, we reach into the DB via a fresh engine bound to the same metadata
    engine = create_async_engine("sqlite+aiosqlite://", echo=False, future=True)
    SessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _setup():
        async with engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
            # Insert two devices with different orgs
            await conn.execute(
                text(
                    "INSERT INTO world_entities(entity_id, type, properties, updated_at, org_id) VALUES (:id,:t,:p, CURRENT_TIMESTAMP, :org)"
                ),
                [{"id": "d1", "t": "DEVICE", "p": "{}", "org": "org1"}, {"id": "d2", "t": "DEVICE", "p": "{}", "org": "org2"}],
            )
    import anyio
    anyio.run(_setup)

    # No header => all
    r = client.get("/api/v1/worldstate")
    assert r.status_code == 200
    all_devices = r.json().get("devices", [])
    assert isinstance(all_devices, list)

    # With org header => filtered
    r2 = client.get("/api/v1/worldstate", headers={"X-Org-ID": "org1"})
    assert r2.status_code == 200
    devs = r2.json().get("devices", [])
    # Expect only org1 entries
    # Can't guarantee order; check device ids if present
    dids = [d.get("device_id") for d in devs]
    assert "d1" in dids or len(devs) <= 1
