"""Deconfliction API — GET /api/v1/deconfliction/conflicts, POST /api/v1/deconfliction/volumes"""

from __future__ import annotations

import sys
import os
import logging
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

# Make packages importable when running inside apps/tasking
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

try:
    from packages.deconfliction.deconfliction_engine import DeconflictionEngine
    from packages.deconfliction.volume import CylinderVolume
except ImportError:
    from deconfliction.deconfliction_engine import DeconflictionEngine  # type: ignore
    from deconfliction.volume import CylinderVolume  # type: ignore

logger = logging.getLogger("tasking.routers.deconfliction")

router = APIRouter(prefix="/api/v1/deconfliction", tags=["deconfliction"])

# ── Engine singleton ──────────────────────────────────────────────────────────

_engine: Optional[DeconflictionEngine] = None


def get_engine() -> DeconflictionEngine:
    """Return (or lazily create) the module-level DeconflictionEngine."""
    global _engine
    if _engine is None:
        _engine = DeconflictionEngine()
        logger.info("DeconflictionEngine initialised (no MQTT)")
    return _engine


# ── Request / Response models ─────────────────────────────────────────────────

class VolumeBody(BaseModel):
    entity_id: str
    lat: float
    lon: float
    radius_m: float = 50.0
    alt_floor_m: float = 0.0
    alt_ceil_m: float = 120.0
    priority: int = 1
    ttl_s: float = 30.0


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/conflicts")
async def list_conflicts():
    """Return all currently active (unresolved) airspace conflicts."""
    engine = get_engine()
    return {"conflicts": engine.get_active_conflicts()}


@router.post("/volumes", status_code=201)
async def register_volume(body: VolumeBody):
    """Register or update an asset's current airspace volume."""
    engine = get_engine()
    vol = CylinderVolume(
        entity_id=body.entity_id,
        lat=body.lat,
        lon=body.lon,
        radius_m=body.radius_m,
        alt_floor_m=body.alt_floor_m,
        alt_ceil_m=body.alt_ceil_m,
        priority=body.priority,
        ttl_s=body.ttl_s,
    )
    engine.update_volume(body.entity_id, vol)
    logger.debug(f"Volume registered for {body.entity_id}")
    return {"status": "ok", "entity_id": body.entity_id}


@router.delete("/volumes/{entity_id}")
async def deregister_volume(entity_id: str):
    """Remove an asset's airspace volume from the deconfliction engine."""
    engine = get_engine()
    engine.remove_volume(entity_id)
    return {"status": "ok", "entity_id": entity_id}


@router.get("/volumes")
async def list_volumes():
    """Return all active (non-stale) airspace volumes."""
    engine = get_engine()
    volumes = [
        {
            "entity_id": v.entity_id,
            "lat": v.lat,
            "lon": v.lon,
            "radius_m": v.radius_m,
            "alt_floor_m": v.alt_floor_m,
            "alt_ceil_m": v.alt_ceil_m,
            "priority": v.priority,
            "ts": v.ts,
            "ttl_s": v.ttl_s,
        }
        for v in engine._active_volumes.values()
        if not v.is_stale()
    ]
    return {"volumes": volumes}
