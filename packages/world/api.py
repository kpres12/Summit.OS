"""
Heli.OS World Model API Router

Mountable FastAPI router providing the canonical entity CRUD endpoints.
Any service can include this router to expose the world model API.

Usage in a FastAPI app:
    from packages.world.api import create_world_router
    from packages.world.store import WorldStore

    world_store = WorldStore()
    app.include_router(create_world_router(world_store), prefix="/api/v1")
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from packages.entities.core import (
    Entity,
    EntityType,
    EntityDomain,
    LifecycleState,
    Kinematics,
    GeoPoint,
    Provenance,
    Relationship,
)
from packages.world.store import WorldStore

logger = logging.getLogger("world.api")


# ── Request / Response Models ──────────────────────────────────


class EntityCreateRequest(BaseModel):
    entity_type: str = "ASSET"
    domain: str = "AERIAL"
    name: str = ""
    class_label: str = ""
    confidence: float = 1.0
    state: str = "ACTIVE"
    # Position (optional)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    heading_deg: float = 0.0
    speed_mps: float = 0.0
    # Provenance
    source_id: str = ""
    source_type: str = ""
    org_id: Optional[str] = None
    # Relationships
    relationships: List[Dict[str, str]] = Field(default_factory=list)
    # Metadata
    metadata: Dict[str, str] = Field(default_factory=dict)
    # TTL
    ttl_seconds: int = 0
    # Domain-specific
    severity: str = ""
    description: str = ""
    mission_status: str = ""
    assigned_asset_ids: List[str] = Field(default_factory=list)


class EntityUpdateRequest(BaseModel):
    name: Optional[str] = None
    class_label: Optional[str] = None
    confidence: Optional[float] = None
    state: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    heading_deg: Optional[float] = None
    speed_mps: Optional[float] = None
    metadata: Optional[Dict[str, str]] = None
    severity: Optional[str] = None
    description: Optional[str] = None
    mission_status: Optional[str] = None
    assigned_asset_ids: Optional[List[str]] = None


class RelationshipRequest(BaseModel):
    related_entity_id: str
    relationship_type: (
        str  # "assigned_to", "observed_by", "correlated_with", "parent", "child"
    )


class BulkUpsertRequest(BaseModel):
    entities: List[EntityCreateRequest]


# ── Router Factory ─────────────────────────────────────────────


def create_world_router(store: WorldStore) -> APIRouter:
    """Create a FastAPI router wired to the given WorldStore."""

    router = APIRouter(tags=["world-model"])

    # ── Entity CRUD ────────────────────────────────────────

    @router.get("/entities")
    async def list_entities(
        entity_type: Optional[str] = None,
        domain: Optional[str] = None,
        state: Optional[str] = None,
        org_id: Optional[str] = None,
        class_label: Optional[str] = None,
        limit: int = Query(default=500, le=5000),
    ):
        """List entities with optional filters."""
        et = EntityType(entity_type) if entity_type else None
        ed = EntityDomain(domain) if domain else None
        ls = LifecycleState(state) if state else None

        entities = store.query(
            entity_type=et,
            domain=ed,
            state=ls,
            org_id=org_id,
            class_label=class_label,
            limit=limit,
        )
        return {
            "entities": [e.to_dict() for e in entities],
            "total": len(entities),
            "version": store.version,
        }

    @router.get("/entities/stats")
    async def world_stats():
        """Get world model statistics."""
        return store.stats()

    @router.get("/entities/nearby")
    async def nearby_entities(
        lat: float,
        lon: float,
        radius_m: float = 1000.0,
        entity_type: Optional[str] = None,
        limit: int = Query(default=100, le=1000),
    ):
        """Find entities within radius of a point."""
        et = EntityType(entity_type) if entity_type else None
        results = store.query_nearby(lat, lon, radius_m, entity_type=et, limit=limit)
        return {
            "entities": [
                {**e.to_dict(), "_distance_m": round(d, 1)} for e, d in results
            ],
            "total": len(results),
        }

    @router.get("/entities/{entity_id}")
    async def get_entity(entity_id: str):
        """Get a single entity by ID."""
        entity = store.get(entity_id)
        if entity is None:
            raise HTTPException(status_code=404, detail="Entity not found")
        return {"entity": entity.to_dict()}

    @router.post("/entities")
    async def create_entity(req: EntityCreateRequest):
        """Create a new entity in the world model."""
        entity = _request_to_entity(req)
        created = store.upsert(entity, source=req.source_type or "api")
        return {"entity": created.to_dict(), "created": True}

    @router.put("/entities/{entity_id}")
    async def update_entity(entity_id: str, req: EntityUpdateRequest):
        """Update an existing entity."""
        entity = store.get(entity_id)
        if entity is None:
            raise HTTPException(status_code=404, detail="Entity not found")

        # Apply updates
        if req.name is not None:
            entity.name = req.name
        if req.class_label is not None:
            entity.class_label = req.class_label
        if req.confidence is not None:
            entity.confidence = req.confidence
        if req.state is not None:
            entity.state = LifecycleState(req.state)
        if req.metadata is not None:
            entity.metadata.update(req.metadata)
        if req.severity is not None:
            entity.severity = req.severity
        if req.description is not None:
            entity.description = req.description
        if req.mission_status is not None:
            entity.mission_status = req.mission_status
        if req.assigned_asset_ids is not None:
            entity.assigned_asset_ids = req.assigned_asset_ids

        # Update position if provided
        if req.latitude is not None or req.longitude is not None:
            if entity.kinematics is None:
                entity.kinematics = Kinematics()
            if entity.kinematics.position is None:
                entity.kinematics.position = GeoPoint()
            if req.latitude is not None:
                entity.kinematics.position.latitude = req.latitude
            if req.longitude is not None:
                entity.kinematics.position.longitude = req.longitude
            if req.altitude is not None:
                entity.kinematics.position.altitude_msl = req.altitude
        if req.heading_deg is not None and entity.kinematics:
            entity.kinematics.heading_deg = req.heading_deg
        if req.speed_mps is not None and entity.kinematics:
            entity.kinematics.speed_mps = req.speed_mps

        updated = store.upsert(entity, source="api")
        return {"entity": updated.to_dict(), "updated": True}

    @router.delete("/entities/{entity_id}")
    async def delete_entity(entity_id: str):
        """Delete an entity from the world model."""
        deleted = store.delete(entity_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Entity not found")
        return {"deleted": True, "entity_id": entity_id}

    @router.post("/entities/bulk")
    async def bulk_upsert(req: BulkUpsertRequest):
        """Bulk upsert entities."""
        entities = [_request_to_entity(r) for r in req.entities]
        count = store.bulk_upsert(entities, source="bulk")
        return {"upserted": count}

    # ── Relationships ──────────────────────────────────────

    @router.get("/entities/{entity_id}/relationships")
    async def get_relationships(entity_id: str):
        """Get all relationships for an entity."""
        entity = store.get(entity_id)
        if entity is None:
            raise HTTPException(status_code=404, detail="Entity not found")
        return {
            "entity_id": entity_id,
            "relationships": [
                {"entity_id": r.entity_id, "relationship": r.relationship}
                for r in entity.relationships
            ],
        }

    @router.post("/entities/{entity_id}/relationships")
    async def add_relationship(entity_id: str, req: RelationshipRequest):
        """Add a relationship to an entity."""
        entity = store.get(entity_id)
        if entity is None:
            raise HTTPException(status_code=404, detail="Entity not found")

        # Verify target exists
        target = store.get(req.related_entity_id)
        if target is None:
            raise HTTPException(status_code=404, detail="Related entity not found")

        # Add relationship (avoid duplicates)
        existing = {(r.entity_id, r.relationship) for r in entity.relationships}
        if (req.related_entity_id, req.relationship_type) not in existing:
            entity.relationships.append(
                Relationship(
                    entity_id=req.related_entity_id,
                    relationship=req.relationship_type,
                )
            )
            store.upsert(entity, source="api")

        return {"added": True, "entity_id": entity_id}

    @router.delete("/entities/{entity_id}/relationships/{related_id}")
    async def remove_relationship(entity_id: str, related_id: str):
        """Remove a relationship from an entity."""
        entity = store.get(entity_id)
        if entity is None:
            raise HTTPException(status_code=404, detail="Entity not found")

        before = len(entity.relationships)
        entity.relationships = [
            r for r in entity.relationships if r.entity_id != related_id
        ]
        if len(entity.relationships) < before:
            store.upsert(entity, source="api")

        return {"removed": before - len(entity.relationships)}

    # ── WebSocket Stream ───────────────────────────────────

    @router.websocket("/entities/stream")
    async def entity_stream(websocket: WebSocket):
        """
        WebSocket endpoint for real-time entity change events.

        Clients receive JSON messages:
        {"event": "create|update|delete", "entity": {...}, "timestamp": ...}

        Optional query params for filtering:
        ?entity_type=TRACK&domain=AERIAL
        """
        await websocket.accept()

        # Parse filters from query params
        params = websocket.query_params
        type_filter = params.get("entity_type")
        domain_filter = params.get("domain")

        queue = store.subscribe()
        try:
            while True:
                event = await queue.get()

                # Apply filters
                if type_filter and event.get("entity_type") != type_filter:
                    continue
                if domain_filter:
                    entity_data = event.get("entity", {})
                    if entity_data.get("domain") != domain_filter:
                        continue

                await websocket.send_json(event)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.debug(f"WebSocket stream error: {e}")
        finally:
            store.unsubscribe(queue)

    # ── Common Operating Picture snapshot ──────────────────

    @router.get("/cop")
    async def common_operating_picture(org_id: Optional[str] = None):
        """
        Get the full Common Operating Picture (COP).

        Returns all active entities organized by type — this is the
        unified singular mission view.
        """
        entities = store.query(org_id=org_id, limit=10000)

        cop: Dict[str, List] = {
            "assets": [],
            "tracks": [],
            "alerts": [],
            "missions": [],
            "sensors": [],
            "geofences": [],
            "observations": [],
            "other": [],
        }

        category_map = {
            EntityType.ASSET: "assets",
            EntityType.TRACK: "tracks",
            EntityType.ALERT: "alerts",
            EntityType.MISSION: "missions",
            EntityType.SENSOR: "sensors",
            EntityType.GEOFENCE: "geofences",
            EntityType.OBSERVATION: "observations",
        }

        for entity in entities:
            category = category_map.get(entity.entity_type, "other")
            cop[category].append(entity.to_dict())

        return {
            "cop": cop,
            "total_entities": len(entities),
            "version": store.version,
            "timestamp": time.time(),
        }

    return router


# ── Helpers ────────────────────────────────────────────────────


def _request_to_entity(req: EntityCreateRequest) -> Entity:
    """Convert API request to Entity dataclass."""
    entity = Entity(
        entity_type=EntityType(req.entity_type),
        domain=EntityDomain(req.domain),
        state=LifecycleState(req.state),
        name=req.name,
        class_label=req.class_label,
        confidence=req.confidence,
        metadata=req.metadata,
        ttl_seconds=req.ttl_seconds,
        severity=req.severity,
        description=req.description,
        mission_status=req.mission_status,
        assigned_asset_ids=req.assigned_asset_ids,
    )

    # Position
    if req.latitude is not None and req.longitude is not None:
        entity.kinematics = Kinematics(
            position=GeoPoint(
                latitude=req.latitude,
                longitude=req.longitude,
                altitude_msl=req.altitude or 0.0,
            ),
            heading_deg=req.heading_deg,
            speed_mps=req.speed_mps,
        )

    # Provenance
    entity.provenance = Provenance(
        source_id=req.source_id,
        source_type=req.source_type,
        org_id=req.org_id or "",
    )

    # Relationships
    for r in req.relationships:
        entity.relationships.append(
            Relationship(
                entity_id=r.get("entity_id", ""),
                relationship=r.get("relationship", ""),
            )
        )

    return entity
