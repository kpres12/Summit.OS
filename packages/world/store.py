"""
Summit.OS World Store — The Single Source of Truth

Every entity in Summit.OS (asset, track, alert, mission, sensor, geofence)
lives in the WorldStore. Services MUST read/write entities through this
interface rather than maintaining their own state.

Architecture:
- In-memory cache (Dict[str, Entity]) for sub-millisecond reads
- Async Postgres persistence (world_entities table) for durability
- Async subscriber queues for real-time entity change notifications
- TTL-based expiration for tracks and observations
- Version tracking for CRDT compatibility

This replaces:
- apps/fabric/main.py's inline world_state dict
- packages/grpc_services/entity_service.py's EntityStore
- Direct world_entities table access across services
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from packages.entities.core import (
    Entity,
    EntityType,
    EntityDomain,
    LifecycleState,
    Provenance,
    Kinematics,
    GeoPoint,
    Relationship,
)

logger = logging.getLogger("world.store")

# ---------------------------------------------------------------------------
# Event types emitted to subscribers
# ---------------------------------------------------------------------------
EVENT_CREATE = "create"
EVENT_UPDATE = "update"
EVENT_DELETE = "delete"
EVENT_EXPIRE = "expire"


class WorldStore:
    """
    Unified world model store.

    Holds every entity in memory for fast reads and persists to Postgres
    asynchronously.  Subscribers receive real-time change events.
    """

    def __init__(self, org_id: str = "default"):
        self.org_id = org_id

        # ── In-memory entity cache ─────────────────────────────
        self._entities: Dict[str, Entity] = {}
        self._version: int = 0

        # ── Subscribers (asyncio Queues) ───────────────────────
        self._subscribers: List[asyncio.Queue] = []

        # ── Optional persistence layer (set via initialize()) ──
        self._engine = None
        self._session_factory = None
        self._persist_enabled = False

        # ── Optional MQTT broadcast ────────────────────────────
        self._mqtt_client = None

        # ── Optional WebSocket manager ─────────────────────────
        self._ws_manager = None

        # ── Callbacks for mesh sync ────────────────────────────
        self._on_change_callbacks: List[Callable] = []

    # ── Initialization ─────────────────────────────────────────

    async def initialize(
        self,
        engine=None,
        session_factory=None,
        mqtt_client=None,
        ws_manager=None,
    ):
        """
        Initialize persistence and broadcast layers.

        Args:
            engine: SQLAlchemy AsyncEngine (optional — runs in-memory only if None)
            session_factory: SQLAlchemy async sessionmaker
            mqtt_client: paho MQTT client for broadcasting entity updates
            ws_manager: WebSocketManager for pushing to console
        """
        self._engine = engine
        self._session_factory = session_factory
        self._mqtt_client = mqtt_client
        self._ws_manager = ws_manager
        self._persist_enabled = engine is not None and session_factory is not None

        if self._persist_enabled:
            await self._ensure_tables()
            await self._load_from_db()

        logger.info(
            f"WorldStore initialized (persist={self._persist_enabled}, "
            f"loaded {len(self._entities)} entities)"
        )

    async def _ensure_tables(self):
        """Create world model tables if they don't exist."""
        from sqlalchemy import text

        async with self._engine.begin() as conn:
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS world_entities_v2 (
                    entity_id   VARCHAR(128) PRIMARY KEY,
                    entity_type VARCHAR(64)  NOT NULL,
                    domain      VARCHAR(32)  NOT NULL DEFAULT 'AERIAL',
                    state       VARCHAR(32)  NOT NULL DEFAULT 'ACTIVE',
                    name        VARCHAR(256),
                    class_label VARCHAR(128),
                    confidence  DOUBLE PRECISION DEFAULT 1.0,
                    properties  JSONB,
                    org_id      VARCHAR(128),
                    version     INTEGER DEFAULT 0,
                    created_at  TIMESTAMPTZ DEFAULT NOW(),
                    updated_at  TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_we2_type ON world_entities_v2 (entity_type)
            """))
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_we2_domain ON world_entities_v2 (domain)
            """))
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_we2_org ON world_entities_v2 (org_id)
            """))
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_we2_state ON world_entities_v2 (state)
            """))

    async def _load_from_db(self):
        """Load active entities from Postgres into memory on startup."""
        from sqlalchemy import text

        async with self._session_factory() as session:
            result = await session.execute(text(
                "SELECT entity_id, entity_type, domain, state, name, class_label, "
                "confidence, properties, org_id, version "
                "FROM world_entities_v2 WHERE state != 'DELETED'"
            ))
            for row in result.all():
                m = dict(row._mapping)
                props = m.get("properties") or {}
                if isinstance(props, str):
                    try:
                        props = json.loads(props)
                    except Exception:
                        props = {}

                entity = Entity(
                    id=m["entity_id"],
                    entity_type=EntityType(m["entity_type"]),
                    domain=EntityDomain(m["domain"]),
                    state=LifecycleState(m["state"]),
                    name=m.get("name") or "",
                    class_label=m.get("class_label") or "",
                    confidence=m.get("confidence") or 1.0,
                    metadata=props.get("metadata", {}),
                )

                # Restore kinematics if present
                if "kinematics" in props and props["kinematics"]:
                    k = props["kinematics"]
                    pos = GeoPoint(**k["position"]) if k.get("position") else None
                    entity.kinematics = Kinematics(
                        position=pos,
                        heading_deg=k.get("heading_deg", 0),
                        speed_mps=k.get("speed_mps", 0),
                    )

                # Restore provenance
                if "provenance" in props and props["provenance"]:
                    entity.provenance = Provenance(**props["provenance"])
                else:
                    entity.provenance = Provenance(
                        org_id=m.get("org_id") or self.org_id,
                        version=m.get("version") or 0,
                    )

                # Restore relationships
                if "relationships" in props and props["relationships"]:
                    entity.relationships = [
                        Relationship(**r) for r in props["relationships"]
                    ]

                self._entities[entity.id] = entity

    # ── Core CRUD ──────────────────────────────────────────────

    def get(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID. Returns None if not found or expired."""
        entity = self._entities.get(entity_id)
        if entity is None:
            return None
        # Check TTL
        if entity.ttl_seconds > 0 and entity.provenance:
            age = time.time() - entity.provenance.updated_at
            if age > entity.ttl_seconds:
                self._expire(entity_id)
                return None
        return entity

    def upsert(self, entity: Entity, source: str = "api") -> Entity:
        """
        Create or update an entity.

        This is the primary write path. Every service should call this
        instead of writing to Postgres directly.
        """
        now = time.time()
        is_new = entity.id not in self._entities

        # Assign ID if missing
        if not entity.id:
            entity.id = str(uuid.uuid4())

        # Bump version
        self._version += 1

        # Ensure provenance
        if entity.provenance is None:
            entity.provenance = Provenance(
                source_id=source,
                source_type=source,
                org_id=self.org_id,
                created_at=now,
                updated_at=now,
                version=self._version,
            )
        else:
            entity.provenance.updated_at = now
            entity.provenance.version = self._version
            if not entity.provenance.org_id:
                entity.provenance.org_id = self.org_id

        # Store
        self._entities[entity.id] = entity

        # Notify
        event_type = EVENT_CREATE if is_new else EVENT_UPDATE
        self._emit(event_type, entity)

        # Persist async (fire-and-forget)
        if self._persist_enabled:
            asyncio.ensure_future(self._persist(entity))

        return entity

    def delete(self, entity_id: str) -> bool:
        """Soft-delete an entity (sets state to DELETED)."""
        entity = self._entities.get(entity_id)
        if entity is None:
            return False

        entity.state = LifecycleState.DELETED
        if entity.provenance:
            entity.provenance.updated_at = time.time()

        self._emit(EVENT_DELETE, entity)

        # Remove from cache
        del self._entities[entity_id]

        if self._persist_enabled:
            asyncio.ensure_future(self._persist_delete(entity_id))

        return True

    def _expire(self, entity_id: str):
        """Handle TTL expiration."""
        entity = self._entities.pop(entity_id, None)
        if entity:
            entity.state = LifecycleState.DELETED
            self._emit(EVENT_EXPIRE, entity)

    # ── Query ──────────────────────────────────────────────────

    def query(
        self,
        entity_type: Optional[EntityType] = None,
        domain: Optional[EntityDomain] = None,
        state: Optional[LifecycleState] = None,
        org_id: Optional[str] = None,
        class_label: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Entity]:
        """Query entities with filters."""
        results = []
        for entity in self._entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue
            if domain and entity.domain != domain:
                continue
            if state and entity.state != state:
                continue
            if org_id and entity.provenance and entity.provenance.org_id != org_id:
                continue
            if class_label and entity.class_label != class_label:
                continue
            # Check TTL
            if entity.ttl_seconds > 0 and entity.provenance:
                age = time.time() - entity.provenance.updated_at
                if age > entity.ttl_seconds:
                    continue
            results.append(entity)
            if len(results) >= limit:
                break
        return results

    def query_nearby(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        entity_type: Optional[EntityType] = None,
        limit: int = 100,
    ) -> List[Tuple[Entity, float]]:
        """
        Query entities within radius_m meters of (lat, lon).

        Returns list of (entity, distance_m) sorted by distance.
        Uses haversine approximation.
        """
        import math

        results = []
        for entity in self._entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue
            if not entity.kinematics or not entity.kinematics.position:
                continue

            pos = entity.kinematics.position
            dlat = math.radians(pos.latitude - lat)
            dlon = math.radians(pos.longitude - lon)
            a = (
                math.sin(dlat / 2) ** 2
                + math.cos(math.radians(lat))
                * math.cos(math.radians(pos.latitude))
                * math.sin(dlon / 2) ** 2
            )
            dist_m = 6371000 * 2 * math.asin(math.sqrt(a))

            if dist_m <= radius_m:
                results.append((entity, dist_m))

        results.sort(key=lambda x: x[1])
        return results[:limit]

    def all_entities(self) -> List[Entity]:
        """Return all non-expired entities."""
        return self.query()

    @property
    def count(self) -> int:
        return len(self._entities)

    @property
    def version(self) -> int:
        return self._version

    # ── Subscriptions (real-time entity stream) ────────────────

    def subscribe(self, max_queue: int = 5000) -> asyncio.Queue:
        """Subscribe to entity change events."""
        q: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        """Unsubscribe from entity change events."""
        self._subscribers = [s for s in self._subscribers if s is not q]

    def on_change(self, callback: Callable):
        """Register a synchronous callback for entity changes (mesh sync)."""
        self._on_change_callbacks.append(callback)

    def _emit(self, event_type: str, entity: Entity):
        """Broadcast entity event to all subscribers and callbacks."""
        event = {
            "event": event_type,
            "entity": entity.to_dict(),
            "entity_id": entity.id,
            "entity_type": entity.entity_type.value,
            "timestamp": time.time(),
            "version": self._version,
        }

        # Async queue subscribers
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass  # Drop if consumer is slow

        # Sync callbacks (mesh, etc.)
        for cb in self._on_change_callbacks:
            try:
                cb(event_type, entity)
            except Exception as e:
                logger.error(f"on_change callback error: {e}")

        # MQTT broadcast
        if self._mqtt_client:
            try:
                topic = f"world/{entity.entity_type.value.lower()}/{entity.id}"
                self._mqtt_client.publish(topic, json.dumps(event), qos=0)
                # Also broadcast to general topic
                self._mqtt_client.publish("world/updates", json.dumps(event), qos=0)
            except Exception as e:
                logger.debug(f"MQTT broadcast failed: {e}")

        # WebSocket broadcast — convert to frontend EntityData format
        if self._ws_manager:
            try:
                ws_msg = self._to_ws_entity_message(event_type, entity)
                asyncio.ensure_future(
                    self._ws_manager.broadcast(json.dumps(ws_msg))
                )
            except Exception:
                pass

    def _to_ws_entity_message(self, event_type: str, entity: Entity) -> dict:
        """Convert Entity to the format expected by the console's useEntityStream hook."""
        pos = entity.kinematics.position if entity.kinematics else None
        # Map EntityType/state to the frontend's entity_type field
        type_map = {
            "ASSET": "friendly",
            "TRACK": "unknown",
            "ALERT": "hostile",
            "OBSERVATION": "neutral",
        }
        fe_type = type_map.get(entity.entity_type.value, "unknown")
        # Map domain
        domain_map = {
            "AERIAL": "aerial",
            "GROUND": "ground",
            "MARITIME": "maritime",
            "FIXED": "fixed",
            "CYBER": "sensor",
        }
        fe_domain = domain_map.get(entity.domain.value, "ground")

        entity_data = {
            "entity_id": entity.id,
            "entity_type": fe_type,
            "domain": fe_domain,
            "classification": entity.class_label,
            "position": {
                "lat": pos.latitude if pos else 0,
                "lon": pos.longitude if pos else 0,
                "alt": pos.altitude_msl if pos else 0,
                "heading_deg": entity.kinematics.heading_deg if entity.kinematics else 0,
            },
            "speed_mps": entity.kinematics.speed_mps if entity.kinematics else 0,
            "confidence": entity.confidence,
            "last_seen": entity.provenance.updated_at if entity.provenance else time.time(),
            "source_sensors": [entity.provenance.source_id] if entity.provenance and entity.provenance.source_id else [],
            "callsign": entity.name or None,
            "battery_pct": entity.aerial.battery_pct if entity.aerial else None,
        }
        if event_type == EVENT_DELETE or event_type == EVENT_EXPIRE:
            return {"type": "entity_removed", "data": {"entity_id": entity.id}}
        return {"type": "entity_update", "data": entity_data}

    # ── Persistence ────────────────────────────────────────────

    async def _persist(self, entity: Entity):
        """Persist entity to Postgres."""
        if not self._session_factory:
            return
        try:
            from sqlalchemy import text

            props = entity.to_dict()
            # Store full entity as properties JSON
            props_json = json.dumps(props, default=str)

            async with self._session_factory() as session:
                await session.execute(text("""
                    INSERT INTO world_entities_v2
                        (entity_id, entity_type, domain, state, name, class_label,
                         confidence, properties, org_id, version, updated_at)
                    VALUES
                        (:eid, :etype, :domain, :state, :name, :label,
                         :conf, :props::jsonb, :org, :ver, NOW())
                    ON CONFLICT (entity_id) DO UPDATE SET
                        entity_type = EXCLUDED.entity_type,
                        domain = EXCLUDED.domain,
                        state = EXCLUDED.state,
                        name = EXCLUDED.name,
                        class_label = EXCLUDED.class_label,
                        confidence = EXCLUDED.confidence,
                        properties = EXCLUDED.properties,
                        org_id = EXCLUDED.org_id,
                        version = EXCLUDED.version,
                        updated_at = NOW()
                """), {
                    "eid": entity.id,
                    "etype": entity.entity_type.value,
                    "domain": entity.domain.value,
                    "state": entity.state.value,
                    "name": entity.name,
                    "label": entity.class_label,
                    "conf": entity.confidence,
                    "props": props_json,
                    "org": entity.provenance.org_id if entity.provenance else self.org_id,
                    "ver": self._version,
                })
                await session.commit()
        except Exception as e:
            logger.error(f"Persist failed for {entity.id}: {e}")

    async def _persist_delete(self, entity_id: str):
        """Mark entity as deleted in Postgres."""
        if not self._session_factory:
            return
        try:
            from sqlalchemy import text

            async with self._session_factory() as session:
                await session.execute(text(
                    "UPDATE world_entities_v2 SET state = 'DELETED', updated_at = NOW() "
                    "WHERE entity_id = :eid"
                ), {"eid": entity_id})
                await session.commit()
        except Exception as e:
            logger.error(f"Persist delete failed for {entity_id}: {e}")

    # ── Bulk operations ────────────────────────────────────────

    def bulk_upsert(self, entities: List[Entity], source: str = "bulk") -> int:
        """Upsert multiple entities. Returns count."""
        count = 0
        for entity in entities:
            self.upsert(entity, source=source)
            count += 1
        return count

    # ── Prune / maintenance ────────────────────────────────────

    def prune_expired(self) -> int:
        """Remove expired entities. Call periodically."""
        now = time.time()
        expired_ids = []
        for eid, entity in self._entities.items():
            if entity.ttl_seconds > 0 and entity.provenance:
                age = now - entity.provenance.updated_at
                if age > entity.ttl_seconds:
                    expired_ids.append(eid)
        for eid in expired_ids:
            self._expire(eid)
        return len(expired_ids)

    # ── Stats ──────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Get world model statistics."""
        by_type: Dict[str, int] = {}
        by_domain: Dict[str, int] = {}
        by_state: Dict[str, int] = {}
        for entity in self._entities.values():
            t = entity.entity_type.value
            d = entity.domain.value
            s = entity.state.value
            by_type[t] = by_type.get(t, 0) + 1
            by_domain[d] = by_domain.get(d, 0) + 1
            by_state[s] = by_state.get(s, 0) + 1

        return {
            "total_entities": len(self._entities),
            "by_type": by_type,
            "by_domain": by_domain,
            "by_state": by_state,
            "version": self._version,
            "subscribers": len(self._subscribers),
            "org_id": self.org_id,
        }

    # ── Snapshot / merge (for mesh sync) ───────────────────────

    def snapshot(self) -> Dict[str, Dict]:
        """
        Return full world state as a dict of entity_id -> entity.to_dict().
        Used for mesh sync digest comparison.
        """
        return {eid: e.to_dict() for eid, e in self._entities.items()}

    def merge_remote(self, entity_dict: Dict[str, Any], source: str = "mesh"):
        """
        Merge a remote entity into the world model.

        Uses last-writer-wins based on provenance.updated_at.
        If remote is newer, it replaces local.
        """
        remote_entity = Entity.from_dict(entity_dict)
        local = self._entities.get(remote_entity.id)

        if local is None:
            # New entity from remote
            self.upsert(remote_entity, source=source)
            return

        # Compare timestamps — last writer wins
        local_ts = local.provenance.updated_at if local.provenance else 0
        remote_ts = 0
        if "provenance" in entity_dict and entity_dict["provenance"]:
            remote_ts = entity_dict["provenance"].get("updated_at", 0)

        if remote_ts > local_ts:
            self.upsert(remote_entity, source=source)
