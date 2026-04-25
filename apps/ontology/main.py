"""
Heli.OS Ontology Service

REST + WebSocket API exposing the Heli.OS Ontology layer.

Endpoints:

  Schema (type definitions — read-only):
    GET  /schema                       — full ontology schema summary
    GET  /schema/objects               — all object type definitions
    GET  /schema/objects/{type_name}   — single object type definition
    GET  /schema/links                 — all link type definitions
    GET  /schema/actions               — all action type definitions

  Objects (live instances):
    GET  /objects/{type_name}          — list instances (with ?filter_* query params)
    GET  /objects/{type_name}/{id}     — get single instance
    POST /objects/{type_name}/sync     — upsert instance (internal sync only)
    DELETE /objects/{type_name}/{id}   — delete instance

  Links:
    GET  /links/{link_type}            — list link instances
    POST /links/{link_type}            — create link
    DELETE /links/{link_type}/{source_id}/{target_id}

  Graph:
    GET  /graph/{type_name}/{id}/neighbors   — all linked objects
    GET  /graph/{type_name}/{id}/traverse    — one-hop traversal

  Actions (governed mutations):
    POST /actions/{action_name}        — execute a governed action
    GET  /audit                        — recent audit trail

  Query:
    POST /query/semantic               — natural-language-friendly semantic query
    GET  /query/summary                — current state summary (for LLM injection)

  Sync helpers (called by other services):
    POST /sync/entity                  — sync raw entity from Fusion
    POST /sync/alert                   — sync raw alert from Intelligence
    POST /sync/observation             — sync raw observation from Intelligence
    POST /sync/mission                 — sync raw mission from Tasking
    POST /sync/sitrep                  — sync raw sitrep from Intelligence

  WebSocket:
    WS   /ws/events                    — stream ontology events (object.created, etc.)

  Health:
    GET  /health
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from packages.ontology import (
    OntologyQuery,
    get_action_runner,
    get_registry,
    get_store,
    get_sync,
    recent_audit,
)
from packages.ontology.types import LinkInstance, ObjectInstance

logger = logging.getLogger("ontology.service")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)

# ── WebSocket connection manager ───────────────────────────────────────────────


class _WSManager:
    def __init__(self):
        self._connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._connections.append(ws)

    def disconnect(self, ws: WebSocket):
        self._connections.discard(ws) if hasattr(self._connections, "discard") else None
        if ws in self._connections:
            self._connections.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self._connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


_ws_manager = _WSManager()


def _broadcast_sync(event: str, payload: Any):
    """Sync wrapper for async broadcast (called from store event listeners)."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(
                _ws_manager.broadcast(
                    {
                        "event": event,
                        "payload": (
                            payload.to_dict()
                            if hasattr(payload, "to_dict")
                            else str(payload)
                        ),
                    }
                )
            )
    except Exception:
        pass


# ── lifespan ───────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Bootstrap ontology (registers all type definitions + indexes)
    registry = get_registry()
    store = get_store()

    # Wire store events → WebSocket broadcast
    store.on("object.created", lambda p: _broadcast_sync("object.created", p))
    store.on("object.updated", lambda p: _broadcast_sync("object.updated", p))
    store.on("object.deleted", lambda p: _broadcast_sync("object.deleted", p))
    store.on("link.created", lambda p: _broadcast_sync("link.created", p))

    summary = registry.summary()
    logger.info(
        "Ontology service ready — %d object types, %d link types, %d action types",
        summary["object_types"],
        summary["link_types"],
        summary["action_types"],
    )
    yield
    logger.info("Ontology service shutting down")


# ── app ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Heli.OS Ontology Service",
    description="Semantic layer and operational backbone for Heli.OS",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request/response models ────────────────────────────────────────────────────


class ActionRequest(BaseModel):
    object_id: str = ""
    inputs: Dict[str, Any] = {}
    actor_id: str = "operator"


class SyncEntityRequest(BaseModel):
    entity: Dict[str, Any]


class SyncAlertRequest(BaseModel):
    alert: Dict[str, Any]
    alert_id: Optional[str] = None


class SyncObservationRequest(BaseModel):
    observation: Dict[str, Any]
    alert_id: Optional[str] = None


class SyncMissionRequest(BaseModel):
    mission: Dict[str, Any]


class SyncSitRepRequest(BaseModel):
    sitrep: Dict[str, Any]


class CreateLinkRequest(BaseModel):
    source_id: str
    target_id: str
    properties: Dict[str, Any] = {}


class SemanticQueryRequest(BaseModel):
    object_type: Optional[str] = None
    filters: Dict[str, Any] = {}
    link_traverse: Optional[str] = None  # link_type to traverse
    traverse_direction: str = "outbound"
    limit: int = 50


# ── health ─────────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    store = get_store()
    reg = get_registry()
    stats = store.stats()
    return {
        "status": "ok",
        "service": "Heli.OS Ontology",
        "schema": reg.summary(),
        "store": stats,
    }


# ── schema endpoints ───────────────────────────────────────────────────────────


@app.get("/schema")
def get_schema():
    return get_registry().summary()


@app.get("/schema/objects")
def list_object_types():
    reg = get_registry()
    return [
        {
            "name": t.name,
            "display_name": t.display_name,
            "description": t.description,
            "properties": [
                {
                    "name": p.name,
                    "kind": p.kind,
                    "required": p.required,
                    "description": p.description,
                    "index": p.index,
                }
                for p in t.properties
            ],
            "links_from": [l.name for l in reg.links_from(t.name)],
            "links_to": [l.name for l in reg.links_to(t.name)],
            "actions": [a.name for a in reg.actions_for(t.name)],
        }
        for t in reg.list_object_types()
    ]


@app.get("/schema/objects/{type_name}")
def get_object_type(type_name: str):
    reg = get_registry()
    defn = reg.get_object_type(type_name)
    if not defn:
        raise HTTPException(404, f"Object type '{type_name}' not found")
    return {
        "name": defn.name,
        "display_name": defn.display_name,
        "description": defn.description,
        "properties": [
            {
                "name": p.name,
                "kind": p.kind,
                "required": p.required,
                "enum_values": p.enum_values,
                "description": p.description,
                "index": p.index,
            }
            for p in defn.properties
        ],
        "links_from": [
            {
                "name": l.name,
                "target_type": l.target_type,
                "cardinality": l.cardinality,
                "description": l.description,
            }
            for l in reg.links_from(type_name)
        ],
        "links_to": [
            {
                "name": l.name,
                "source_type": l.source_type,
                "cardinality": l.cardinality,
                "description": l.description,
            }
            for l in reg.links_to(type_name)
        ],
        "actions": [
            {
                "name": a.name,
                "display_name": a.display_name,
                "description": a.description,
                "inputs": [
                    {"name": p.name, "kind": p.kind, "required": p.required}
                    for p in a.input_properties
                ],
            }
            for a in reg.actions_for(type_name)
        ],
    }


@app.get("/schema/links")
def list_link_types():
    return [
        {
            "name": l.name,
            "display_name": l.display_name,
            "source_type": l.source_type,
            "target_type": l.target_type,
            "cardinality": l.cardinality,
            "description": l.description,
        }
        for l in get_registry().list_link_types()
    ]


@app.get("/schema/actions")
def list_action_types():
    return [
        {
            "name": a.name,
            "display_name": a.display_name,
            "target_type": a.target_type,
            "description": a.description,
            "inputs": [
                {
                    "name": p.name,
                    "kind": p.kind,
                    "required": p.required,
                    "enum_values": p.enum_values,
                }
                for p in a.input_properties
            ],
        }
        for a in get_registry().list_action_types()
    ]


# ── object endpoints ───────────────────────────────────────────────────────────


@app.get("/objects/{type_name}")
def list_objects(
    type_name: str,
    limit: int = Query(100, le=1000),
    offset: int = Query(0),
    status: Optional[str] = None,
    severity: Optional[str] = None,
    domain: Optional[str] = None,
    org_id: Optional[str] = None,
):
    reg = get_registry()
    if not reg.get_object_type(type_name):
        raise HTTPException(404, f"Object type '{type_name}' not found")

    filters = {}
    if status:
        filters["status"] = status
    if severity:
        filters["severity"] = severity
    if domain:
        filters["domain"] = domain
    if org_id:
        filters["org_id"] = org_id

    instances = get_store().list(type_name, filters or None, limit, offset)
    return {
        "type": type_name,
        "count": len(instances),
        "objects": [i.to_dict() for i in instances],
    }


@app.get("/objects/{type_name}/{object_id}")
def get_object(type_name: str, object_id: str):
    instance = get_store().get(type_name, object_id)
    if not instance:
        raise HTTPException(404, f"{type_name} '{object_id}' not found")
    return instance.to_dict()


@app.post("/objects/{type_name}/sync")
def sync_object(type_name: str, body: Dict[str, Any]):
    """Internal: direct upsert without going through ActionRunner."""
    reg = get_registry()
    if not reg.get_object_type(type_name):
        raise HTTPException(404, f"Object type '{type_name}' not found")
    obj_id = body.get("id", body.get("object_id", ""))
    if not obj_id:
        raise HTTPException(400, "Missing 'id' in body")
    instance = ObjectInstance(object_type=type_name, object_id=obj_id, properties=body)
    result = get_store()._upsert(instance)
    return result.to_dict()


@app.delete("/objects/{type_name}/{object_id}")
def delete_object(type_name: str, object_id: str):
    deleted = get_store().delete(type_name, object_id)
    if not deleted:
        raise HTTPException(404, f"{type_name} '{object_id}' not found")
    return {"deleted": True, "object_id": object_id}


# ── link endpoints ─────────────────────────────────────────────────────────────


@app.get("/links/{link_type}")
def list_links(
    link_type: str, source_id: Optional[str] = None, target_id: Optional[str] = None
):
    store = get_store()
    if source_id:
        links = store.links_from_object(source_id, link_type)
    elif target_id:
        links = store.links_to_object(target_id, link_type)
    else:
        links = [l for (lt, _, _), l in store._links.items() if lt == link_type]
    return {
        "link_type": link_type,
        "count": len(links),
        "links": [l.to_dict() for l in links],
    }


@app.post("/links/{link_type}")
def create_link(link_type: str, body: CreateLinkRequest):
    reg = get_registry()
    if not reg.get_link_type(link_type):
        raise HTTPException(404, f"Link type '{link_type}' not found")
    link = LinkInstance(
        link_type=link_type,
        source_id=body.source_id,
        target_id=body.target_id,
        properties=body.properties,
    )
    result = get_store()._upsert_link(link)
    return result.to_dict()


@app.delete("/links/{link_type}/{source_id}/{target_id}")
def delete_link(link_type: str, source_id: str, target_id: str):
    deleted = get_store().delete_link(link_type, source_id, target_id)
    if not deleted:
        raise HTTPException(404, "Link not found")
    return {"deleted": True}


# ── graph endpoints ────────────────────────────────────────────────────────────


@app.get("/graph/{type_name}/{object_id}/neighbors")
def get_neighbors(type_name: str, object_id: str):
    instance = get_store().get(type_name, object_id)
    if not instance:
        raise HTTPException(404, f"{type_name} '{object_id}' not found")
    q = OntologyQuery()
    neighbors = q.neighbors(object_id)
    return {
        "object_id": object_id,
        "object_type": type_name,
        "neighbors": {
            rel: [n.to_dict() for n in nodes] for rel, nodes in neighbors.items()
        },
    }


@app.get("/graph/{type_name}/{object_id}/traverse")
def traverse(
    type_name: str,
    object_id: str,
    link_type: str,
    direction: str = "outbound",
    target_type: Optional[str] = None,
):
    q = OntologyQuery()
    results = q.traverse(type_name, object_id, link_type, direction, target_type)
    return {
        "source_id": object_id,
        "link_type": link_type,
        "direction": direction,
        "count": len(results),
        "objects": [r.to_dict() for r in results],
    }


# ── action endpoints ───────────────────────────────────────────────────────────


@app.post("/actions/{action_name}")
def execute_action(action_name: str, body: ActionRequest):
    reg = get_registry()
    if not reg.get_action(action_name):
        raise HTTPException(404, f"Action '{action_name}' not found")

    result = get_action_runner().execute(
        action_name=action_name,
        object_id=body.object_id,
        inputs=body.inputs,
        actor_id=body.actor_id,
    )

    if not result.success:
        raise HTTPException(
            422,
            detail={
                "error": result.error,
                "audit": result.audit_entry.to_dict() if result.audit_entry else None,
            },
        )

    return {
        "success": True,
        "object": result.object_instance.to_dict() if result.object_instance else None,
        "audit": result.audit_entry.to_dict() if result.audit_entry else None,
        "side_effect_log": result.side_effect_log,
    }


@app.get("/audit")
def get_audit(limit: int = Query(100, le=1000), actor_id: Optional[str] = None):
    entries = recent_audit(limit=limit, actor_id=actor_id)
    return {"count": len(entries), "entries": [e.to_dict() for e in entries]}


# ── query endpoints ────────────────────────────────────────────────────────────


@app.post("/query/semantic")
def semantic_query(body: SemanticQueryRequest):
    q = OntologyQuery()
    if body.object_type:
        query_builder = q.objects(body.object_type)
        if body.filters:
            query_builder.where(**body.filters)
        query_builder.limit(body.limit)
        results = query_builder.all()
        return {"count": len(results), "objects": [r.to_dict() for r in results]}
    # No type specified → summary
    return {"summary": q.semantic_summary()}


@app.get("/query/summary")
def ontology_summary():
    return {"summary": OntologyQuery().semantic_summary()}


# ── sync endpoints (called by other services) ──────────────────────────────────


@app.post("/sync/entity")
def sync_entity(body: SyncEntityRequest):
    result = get_sync().from_entity(body.entity)
    return result.to_dict()


@app.post("/sync/alert")
def sync_alert(body: SyncAlertRequest):
    result = get_sync().from_alert(body.alert)
    return result.to_dict()


@app.post("/sync/observation")
def sync_observation(body: SyncObservationRequest):
    result = get_sync().from_observation(body.observation, body.alert_id)
    return result.to_dict()


@app.post("/sync/mission")
def sync_mission(body: SyncMissionRequest):
    result = get_sync().from_mission(body.mission)
    return result.to_dict()


@app.post("/sync/sitrep")
def sync_sitrep(body: SyncSitRepRequest):
    result = get_sync().from_sitrep(body.sitrep)
    return result.to_dict()


# ── WebSocket ──────────────────────────────────────────────────────────────────


@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    await _ws_manager.connect(websocket)
    try:
        # Send current state on connect
        await websocket.send_json(
            {
                "event": "connected",
                "payload": get_registry().summary(),
            }
        )
        while True:
            await websocket.receive_text()  # keep connection alive (client ping)
    except WebSocketDisconnect:
        _ws_manager.disconnect(websocket)


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("ONTOLOGY_HOST", "0.0.0.0"),
        port=int(os.getenv("ONTOLOGY_PORT", "8007")),
        reload=os.getenv("ENV", "production") == "development",
        workers=1,
    )
