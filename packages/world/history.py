"""
Entity Position History — ring-buffer position trails for the WorldStore.

Each entity keeps a rolling window of timestamped positions. The console
can request a trail (polyline) for map visualization. Runs entirely in-memory
with optional Postgres persistence.

API routes are registered on the FastAPI app by calling `register_routes(app, store)`.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Deque, Dict, List, Optional

logger = logging.getLogger("summit.world.history")

# Default ring-buffer depth per entity
DEFAULT_MAX_POINTS = int(500)

# Minimum distance (meters) between consecutive recorded points — avoids
# flooding history with stationary-asset noise.
DEFAULT_MIN_DISTANCE_M = 2.0


@dataclass
class PositionSample:
    ts: float          # Unix epoch seconds
    lat: float
    lon: float
    alt: Optional[float] = None
    speed: Optional[float] = None
    heading: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Drop None fields
        return {k: v for k, v in d.items() if v is not None}


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    R = 6_371_000.0
    import math
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class EntityHistory:
    """In-memory position trail for one entity."""

    def __init__(
        self,
        entity_id: str,
        max_points: int = DEFAULT_MAX_POINTS,
        min_distance_m: float = DEFAULT_MIN_DISTANCE_M,
    ):
        self.entity_id = entity_id
        self.max_points = max_points
        self.min_distance_m = min_distance_m
        self._trail: Deque[PositionSample] = deque(maxlen=max_points)

    def record(self, lat: float, lon: float, alt: Optional[float] = None,
               speed: Optional[float] = None, heading: Optional[float] = None,
               ts: Optional[float] = None) -> bool:
        """Record a position. Returns True if a new point was added."""
        if ts is None:
            ts = time.time()

        # Skip if too close to last recorded point (stationary asset)
        if self._trail:
            last = self._trail[-1]
            dist = _haversine_m(last.lat, last.lon, lat, lon)
            if dist < self.min_distance_m:
                return False

        self._trail.append(PositionSample(
            ts=ts, lat=lat, lon=lon, alt=alt,
            speed=speed, heading=heading,
        ))
        return True

    def trail(self, limit: Optional[int] = None) -> List[dict]:
        pts = list(self._trail)
        if limit and limit > 0:
            pts = pts[-limit:]
        return [p.to_dict() for p in pts]

    def clear(self) -> None:
        self._trail.clear()

    def __len__(self) -> int:
        return len(self._trail)


class HistoryStore:
    """Registry of per-entity history trails."""

    def __init__(
        self,
        max_points: int = DEFAULT_MAX_POINTS,
        min_distance_m: float = DEFAULT_MIN_DISTANCE_M,
    ):
        self.max_points = max_points
        self.min_distance_m = min_distance_m
        self._histories: Dict[str, EntityHistory] = {}

    def _get_or_create(self, entity_id: str) -> EntityHistory:
        if entity_id not in self._histories:
            self._histories[entity_id] = EntityHistory(
                entity_id=entity_id,
                max_points=self.max_points,
                min_distance_m=self.min_distance_m,
            )
        return self._histories[entity_id]

    def record(self, entity_id: str, lat: float, lon: float,
               alt: Optional[float] = None, speed: Optional[float] = None,
               heading: Optional[float] = None, ts: Optional[float] = None) -> bool:
        return self._get_or_create(entity_id).record(lat, lon, alt, speed, heading, ts)

    def record_from_entity(self, entity: dict) -> bool:
        """Extract position from an entity dict and record it."""
        entity_id = entity.get("entity_id")
        if not entity_id:
            return False
        kin = entity.get("kinematics") or {}
        pos = kin.get("position") or {}
        lat = pos.get("latitude")
        lon = pos.get("longitude")
        if lat is None or lon is None:
            return False
        alt = pos.get("altitude")
        vel = kin.get("velocity") or {}
        speed = vel.get("speed")
        heading = vel.get("heading")
        ts = entity.get("ts") or entity.get("updated_at")
        if isinstance(ts, str):
            try:
                from datetime import datetime
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
            except Exception:
                ts = None
        return self.record(entity_id, lat, lon, alt, speed, heading, ts)

    def trail(self, entity_id: str, limit: Optional[int] = None) -> List[dict]:
        hist = self._histories.get(entity_id)
        if hist is None:
            return []
        return hist.trail(limit)

    def evict(self, entity_id: str) -> None:
        self._histories.pop(entity_id, None)

    def entity_ids(self) -> List[str]:
        return list(self._histories.keys())

    def summary(self) -> dict:
        return {
            "entity_count": len(self._histories),
            "total_points": sum(len(h) for h in self._histories.values()),
        }


# ── FastAPI route registration ────────────────────────────────────────────────

def register_routes(app, history_store: HistoryStore) -> None:
    """Mount history endpoints onto an existing FastAPI app."""
    from fastapi import HTTPException, Query

    @app.get("/entities/{entity_id}/trail")
    async def get_entity_trail(
        entity_id: str,
        limit: Optional[int] = Query(default=None, ge=1, le=DEFAULT_MAX_POINTS),
    ):
        trail = history_store.trail(entity_id, limit=limit)
        return {
            "entity_id": entity_id,
            "count": len(trail),
            "trail": trail,
        }

    @app.get("/history/summary")
    async def get_history_summary():
        return history_store.summary()

    @app.delete("/entities/{entity_id}/trail")
    async def clear_entity_trail(entity_id: str):
        hist = history_store._histories.get(entity_id)
        if hist is None:
            raise HTTPException(status_code=404, detail="Entity not found in history")
        hist.clear()
        return {"entity_id": entity_id, "cleared": True}
