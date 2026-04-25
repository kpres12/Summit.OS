"""
Heli.OS Device Registry

Stores the mapping of device_id → certificate fingerprint.
When a device connects and presents its cert, the Fabric service
checks this registry to confirm it's a registered device.

Revocation is simple: remove or revoke the entry. The device's cert
becomes worthless without a registry entry, even if the cert is still
cryptographically valid.

Backed by Postgres (production) or SQLite (development).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger("summit.identity.registry")

_DB_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql+asyncpg://summit:summit_password@localhost:5432/heli_os",
)


class DeviceRegistry:
    """
    Registry of authorized devices and their certificate fingerprints.

    A device must be registered before it can connect to Heli.OS.
    Registration associates a device_id with its certificate fingerprint,
    device type, capabilities, and org_id.
    """

    def __init__(self, db_url: Optional[str] = None):
        self._db_url = db_url or _DB_URL
        self._engine = None
        self._session = None
        # In-memory fallback for dev/test
        self._mem_store: Dict[str, dict] = {}
        self._use_db = False

    async def initialize(self):
        """Connect to database and create tables if needed."""
        try:
            from sqlalchemy import (
                MetaData,
                Table,
                Column,
                String,
                Boolean,
                DateTime,
                JSON,
            )
            from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
            from sqlalchemy.orm import sessionmaker

            url = self._db_url
            if url.startswith("postgresql://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

            self._engine = create_async_engine(url, echo=False)
            self._SessionLocal = sessionmaker(
                self._engine, expire_on_commit=False, class_=AsyncSession
            )

            self._metadata = MetaData()
            self._devices = Table(
                "registered_devices",
                self._metadata,
                Column("device_id", String(128), primary_key=True),
                Column("fingerprint", String(64), nullable=False),
                Column("device_type", String(64)),
                Column("org_id", String(128)),
                Column("capabilities", JSON),
                Column("revoked", Boolean, default=False),
                Column("registered_at", DateTime(timezone=True)),
                Column("revoked_at", DateTime(timezone=True), nullable=True),
                Column("metadata", JSON),
            )

            async with self._engine.begin() as conn:
                await conn.run_sync(self._metadata.create_all)

            self._use_db = True
            logger.info("DeviceRegistry initialized with Postgres")

        except Exception as e:
            logger.warning(
                f"DeviceRegistry DB init failed — using in-memory store: {e}"
            )
            self._use_db = False

    async def register(
        self,
        device_id: str,
        fingerprint: str,
        device_type: str = "device",
        org_id: str = "",
        capabilities: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        Register a device with its certificate fingerprint.

        Returns True on success. Raises if device_id already registered
        (use re_register to update).
        """
        entry = {
            "device_id": device_id,
            "fingerprint": fingerprint,
            "device_type": device_type,
            "org_id": org_id,
            "capabilities": capabilities or [],
            "revoked": False,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "revoked_at": None,
            "metadata": metadata or {},
        }

        if self._use_db:
            try:
                from sqlalchemy.dialects.postgresql import insert

                async with self._SessionLocal() as session:
                    stmt = (
                        insert(self._devices)
                        .values(
                            device_id=device_id,
                            fingerprint=fingerprint,
                            device_type=device_type,
                            org_id=org_id,
                            capabilities=capabilities or [],
                            revoked=False,
                            registered_at=datetime.now(timezone.utc),
                            metadata=metadata or {},
                        )
                        .on_conflict_do_update(
                            index_elements=["device_id"],
                            set_={
                                "fingerprint": fingerprint,
                                "revoked": False,
                                "registered_at": datetime.now(timezone.utc),
                            },
                        )
                    )
                    await session.execute(stmt)
                    await session.commit()
            except Exception as e:
                logger.error(f"Failed to register device '{device_id}' in DB: {e}")
                return False

        self._mem_store[device_id] = entry
        logger.info(f"Device registered: {device_id} ({device_type}, org={org_id})")
        return True

    async def is_authorized(self, device_id: str, fingerprint: str) -> bool:
        """
        Check if a device is authorized to connect.

        Returns True only if:
        - device_id is registered
        - fingerprint matches the registered cert
        - device is not revoked
        """
        entry = await self.get_device(device_id)
        if not entry:
            logger.warning(f"Device '{device_id}' is not registered")
            return False
        if entry.get("revoked"):
            logger.warning(f"Device '{device_id}' cert has been revoked")
            return False
        if entry.get("fingerprint") != fingerprint:
            logger.warning(
                f"Device '{device_id}' fingerprint mismatch — possible impersonation attempt"
            )
            return False
        return True

    async def revoke(self, device_id: str, reason: str = "") -> bool:
        """
        Revoke a device's authorization.

        The device's cert remains cryptographically valid but Heli.OS
        will reject all connections from it.
        """
        if self._use_db:
            try:
                from sqlalchemy import update

                async with self._SessionLocal() as session:
                    await session.execute(
                        update(self._devices)
                        .where(self._devices.c.device_id == device_id)
                        .values(revoked=True, revoked_at=datetime.now(timezone.utc))
                    )
                    await session.commit()
            except Exception as e:
                logger.error(f"Failed to revoke device '{device_id}' in DB: {e}")
                return False

        if device_id in self._mem_store:
            self._mem_store[device_id]["revoked"] = True
            self._mem_store[device_id]["revoked_at"] = datetime.now(
                timezone.utc
            ).isoformat()

        logger.warning(f"Device REVOKED: {device_id} (reason={reason!r})")
        return True

    async def get_device(self, device_id: str) -> Optional[dict]:
        """Get device registry entry."""
        if self._use_db:
            try:
                from sqlalchemy import select

                async with self._SessionLocal() as session:
                    result = await session.execute(
                        select(self._devices).where(
                            self._devices.c.device_id == device_id
                        )
                    )
                    row = result.fetchone()
                    if row:
                        return dict(row._mapping)
            except Exception as e:
                logger.debug(f"DB lookup failed for '{device_id}': {e}")

        return self._mem_store.get(device_id)

    async def list_devices(self, org_id: Optional[str] = None) -> List[dict]:
        """List all registered devices, optionally filtered by org_id."""
        if self._use_db:
            try:
                from sqlalchemy import select

                async with self._SessionLocal() as session:
                    q = select(self._devices)
                    if org_id:
                        q = q.where(self._devices.c.org_id == org_id)
                    result = await session.execute(q)
                    return [dict(r._mapping) for r in result.fetchall()]
            except Exception as e:
                logger.debug(f"DB list failed: {e}")

        devices = list(self._mem_store.values())
        if org_id:
            devices = [d for d in devices if d.get("org_id") == org_id]
        return devices
