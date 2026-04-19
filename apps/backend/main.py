#!/usr/bin/env python3
"""
Heli.OS Backend — single-file FastAPI service.

Provides everything the console needs:
  - WebSocket entity stream  ws://localhost:8001/ws
  - REST API                 http://localhost:8001/v1/...

Entities are simulated and move realistically. Geofences and missions
are persisted in SQLite (summit_backend.db in the working directory).

Usage:
    pip install fastapi uvicorn websockets
    python apps/backend/main.py
"""

import asyncio
import json
import math
import os
import random
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

PORT = int(os.getenv("BACKEND_PORT", "8002"))
DB_PATH = os.getenv("BACKEND_DB", "summit_backend.db")
TICK_INTERVAL = 1.0  # seconds between entity updates broadcast to clients

# ──────────────────────────────────────────────────────────────────────────────
# SQLite persistence
# ──────────────────────────────────────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS geofences (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            type        TEXT DEFAULT 'exclusion',
            coordinates TEXT NOT NULL,
            active      INTEGER DEFAULT 1,
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS missions (
            mission_id   TEXT PRIMARY KEY,
            name         TEXT,
            mission_type TEXT DEFAULT 'SURVEY',
            objectives   TEXT DEFAULT '[]',
            status       TEXT DEFAULT 'ACTIVE',
            priority     TEXT DEFAULT 'MEDIUM',
            pattern      TEXT DEFAULT 'grid',
            altitude_m   INTEGER DEFAULT 120,
            area         TEXT,
            asset_ids    TEXT DEFAULT '[]',
            created_at   TEXT NOT NULL,
            started_at   TEXT,
            completed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS alerts (
            alert_id    TEXT PRIMARY KEY,
            severity    TEXT NOT NULL,
            description TEXT NOT NULL,
            source      TEXT NOT NULL,
            ts_iso      TEXT NOT NULL,
            acknowledged INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS entity_trail (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id   TEXT NOT NULL,
            lat         REAL NOT NULL,
            lon         REAL NOT NULL,
            ts          REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_trail_entity ON entity_trail(entity_id, ts);
    """)
    conn.commit()
    conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# Entity simulation
# ──────────────────────────────────────────────────────────────────────────────

ENTITY_TEMPLATES = [
    # UAV fleet — active missions
    {"entity_id": "echo-1", "callsign": "ECHO-1", "entity_type": "active", "domain": "aerial", "classification": "Quadrotor UAV", "speed_mps": 12.0, "battery_pct": 87.0},
    {"entity_id": "echo-2", "callsign": "ECHO-2", "entity_type": "active", "domain": "aerial", "classification": "Quadrotor UAV", "speed_mps": 9.5,  "battery_pct": 62.0},
    {"entity_id": "echo-3", "callsign": "ECHO-3", "entity_type": "active", "domain": "aerial", "classification": "Fixed-wing UAV", "speed_mps": 22.0, "battery_pct": 45.0},
    {"entity_id": "delta-1","callsign": "DELTA-1","entity_type": "active", "domain": "aerial", "classification": "Quadrotor UAV", "speed_mps": 8.0,  "battery_pct": 91.0},
    # ADS-B commercial tracks
    {"entity_id": "swa2013","callsign": "SWA2013","entity_type": "neutral","domain": "aerial","classification": "Boeing 737",   "speed_mps": 220.0,"battery_pct": None},
    {"entity_id": "dal1875","callsign": "DAL1875","entity_type": "neutral","domain": "aerial","classification": "Airbus A320",  "speed_mps": 235.0,"battery_pct": None},
    {"entity_id": "aal512", "callsign": "AAL512", "entity_type": "neutral","domain": "aerial","classification": "Boeing 787",   "speed_mps": 245.0,"battery_pct": None},
    # Ground vehicles
    {"entity_id": "rover-1","callsign": "ROVER-1","entity_type": "active","domain": "ground","classification": "Ground Robot",  "speed_mps": 1.8,  "battery_pct": 76.0},
    {"entity_id": "rover-2","callsign": "ROVER-2","entity_type": "active","domain": "ground","classification": "Ground Robot",  "speed_mps": 2.1,  "battery_pct": 34.0},
    # Alert entity
    {"entity_id": "unk-7f2a","callsign": None,    "entity_type": "alert", "domain": "aerial","classification": "Unknown UAS",  "speed_mps": 18.0, "battery_pct": None},
    # Sensor nodes (fixed)
    {"entity_id": "sensor-a","callsign": "RADAR-A","entity_type": "active","domain": "sensor","classification": "Radar Node",   "speed_mps": 0.0,  "battery_pct": None},
    {"entity_id": "sensor-b","callsign": "RADAR-B","entity_type": "active","domain": "sensor","classification": "RF Sensor",    "speed_mps": 0.0,  "battery_pct": None},
]

# Seed starting positions around the continental US center
SEED_POSITIONS = {
    "echo-1":  (37.82, -122.48),  # San Francisco bay
    "echo-2":  (37.77, -122.41),
    "echo-3":  (37.80, -122.45),
    "delta-1": (37.86, -122.29),
    "swa2013": (38.50, -100.0),   # Cross-country
    "dal1875": (40.10, -104.0),
    "aal512":  (35.20, -118.0),
    "rover-1": (37.78, -122.39),
    "rover-2": (37.77, -122.37),
    "unk-7f2a":(37.91, -122.52),
    "sensor-a":(37.79, -122.44),
    "sensor-b":(37.83, -122.42),
}

# Heading state per entity (radians)
_entity_headings: dict[str, float] = {eid: random.uniform(0, 2 * math.pi) for eid in SEED_POSITIONS}
_entity_positions: dict[str, tuple[float, float]] = dict(SEED_POSITIONS)
_entity_batteries: dict[str, float] = {
    e["entity_id"]: e["battery_pct"]
    for e in ENTITY_TEMPLATES
    if e["battery_pct"] is not None
}


def _tick_positions() -> None:
    """Advance each entity's simulated position by one tick."""
    for tmpl in ENTITY_TEMPLATES:
        eid = tmpl["entity_id"]
        speed = tmpl["speed_mps"]
        if speed == 0:
            continue  # fixed sensors don't move

        lat, lon = _entity_positions[eid]
        hdg = _entity_headings[eid]

        # Random walk — small heading change each tick
        hdg += random.gauss(0, 0.04)
        _entity_headings[eid] = hdg

        # Convert speed to degrees (approximate, good enough for simulation)
        d_deg = speed * TICK_INTERVAL / 111_000  # ~111km per degree lat
        lat += d_deg * math.cos(hdg)
        lon += d_deg * math.sin(hdg) / max(math.cos(math.radians(lat)), 0.01)

        # Soft boundary — nudge back toward seed if wandering too far
        seed_lat, seed_lon = SEED_POSITIONS[eid]
        dlat = seed_lat - lat
        dlon = seed_lon - lon
        dist = math.sqrt(dlat ** 2 + dlon ** 2)
        MAX_WANDER = 0.5  # degrees (~55km)
        if dist > MAX_WANDER:
            # Turn toward seed
            toward = math.atan2(dlon, dlat)
            _entity_headings[eid] = toward + random.gauss(0, 0.2)

        _entity_positions[eid] = (lat, lon)

    # Drain batteries slowly
    for eid in list(_entity_batteries.keys()):
        _entity_batteries[eid] = max(0.0, _entity_batteries[eid] - random.uniform(0.01, 0.04))


def _build_entity_update(tmpl: dict) -> dict:
    eid = tmpl["entity_id"]
    lat, lon = _entity_positions[eid]
    hdg = math.degrees(_entity_headings[eid]) % 360

    entity: dict[str, Any] = {
        "entity_id": eid,
        "entity_type": tmpl["entity_type"],
        "domain": tmpl["domain"],
        "classification": tmpl["classification"],
        "position": {
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "alt": round(random.uniform(50, 3500) if tmpl["domain"] == "aerial" and tmpl["speed_mps"] > 5 else 10, 1),
            "heading_deg": round(hdg, 1),
        },
        "speed_mps": round(tmpl["speed_mps"] + random.gauss(0, 0.5), 2),
        "confidence": round(random.uniform(0.82, 0.99), 2),
        "last_seen": time.time(),
        "source_sensors": ["ADS-B"] if tmpl["domain"] == "aerial" and tmpl["speed_mps"] > 50 else ["MQTT"],
        "track_state": "confirmed",
    }
    if tmpl["callsign"]:
        entity["callsign"] = tmpl["callsign"]
    if eid in _entity_batteries:
        entity["battery_pct"] = round(_entity_batteries[eid], 1)
    return entity


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket connection manager
# ──────────────────────────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self._clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._clients:
            self._clients.remove(ws)

    async def broadcast(self, message: dict) -> None:
        dead = []
        for ws in self._clients:
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def count(self) -> int:
        return len(self._clients)


manager = ConnectionManager()

# ──────────────────────────────────────────────────────────────────────────────
# Background tick task
# ──────────────────────────────────────────────────────────────────────────────

async def entity_broadcast_loop() -> None:
    """Move entities and broadcast updates every TICK_INTERVAL seconds."""
    conn = get_db()
    tick = 0
    while True:
        await asyncio.sleep(TICK_INTERVAL)
        tick += 1

        _tick_positions()

        # Build batch
        batch = [_build_entity_update(t) for t in ENTITY_TEMPLATES]

        # Persist trail every 10 ticks (10s) — keep last 200 points per entity
        if tick % 10 == 0:
            for e in batch:
                conn.execute(
                    "INSERT INTO entity_trail (entity_id, lat, lon, ts) VALUES (?, ?, ?, ?)",
                    (e["entity_id"], e["position"]["lat"], e["position"]["lon"], e["last_seen"]),
                )
                # Trim old trail points
                conn.execute("""
                    DELETE FROM entity_trail WHERE id IN (
                        SELECT id FROM entity_trail WHERE entity_id=? ORDER BY ts ASC
                        LIMIT MAX(0, (SELECT COUNT(*) FROM entity_trail WHERE entity_id=?) - 200)
                    )
                """, (e["entity_id"], e["entity_id"]))
            conn.commit()

        # Broadcast if anyone is connected
        if manager.count > 0:
            await manager.broadcast({"type": "entity_batch", "data": batch})

        # Periodically generate a random alert
        if tick % 45 == 0:
            alert_entity = random.choice([t for t in ENTITY_TEMPLATES if t["entity_type"] == "alert"])
            alert_id = f"ALT-{uuid.uuid4().hex[:8].upper()}"
            severity = random.choice(["HIGH", "CRITICAL"])
            ts_iso = datetime.now(timezone.utc).isoformat()
            description = f"Anomalous behavior detected: {alert_entity.get('callsign') or alert_entity['entity_id']}"
            conn.execute(
                "INSERT OR IGNORE INTO alerts (alert_id, severity, description, source, ts_iso) VALUES (?,?,?,?,?)",
                (alert_id, severity, description, alert_entity["entity_id"], ts_iso),
            )
            conn.commit()


# ──────────────────────────────────────────────────────────────────────────────
# App lifespan
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    task = asyncio.create_task(entity_broadcast_loop())
    print(f"\n  Heli.OS Backend  →  http://localhost:{PORT}")
    print(f"  WebSocket stream   →  ws://localhost:{PORT}/ws")
    print(f"  REST API           →  http://localhost:{PORT}/v1/\n")
    yield
    task.cancel()


app = FastAPI(title="Heli.OS Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ──────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        # Send current entity state immediately on connect
        batch = [_build_entity_update(t) for t in ENTITY_TEMPLATES]
        await ws.send_text(json.dumps({"type": "entity_batch", "data": batch}))

        # Handle pings and subscribe messages
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=35.0)
                msg = json.loads(raw)
                if msg.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
            except asyncio.TimeoutError:
                # Send keepalive
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(ws)


# ──────────────────────────────────────────────────────────────────────────────
# REST — Entities
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/v1/entities/{entity_id}/trail")
async def get_entity_trail(entity_id: str):
    conn = get_db()
    rows = conn.execute(
        "SELECT lat, lon FROM entity_trail WHERE entity_id=? ORDER BY ts DESC LIMIT 100",
        (entity_id,),
    ).fetchall()
    conn.close()
    # Return in chronological order
    trail = [{"lat": r["lat"], "lon": r["lon"]} for r in reversed(rows)]
    # If no DB trail yet, return last simulated position
    if not trail and entity_id in _entity_positions:
        lat, lon = _entity_positions[entity_id]
        trail = [{"lat": lat, "lon": lon}]
    return {"entity_id": entity_id, "trail": trail}


# ──────────────────────────────────────────────────────────────────────────────
# REST — Alerts
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/v1/alerts")
async def list_alerts(limit: int = 100):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM alerts ORDER BY ts_iso DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return {"alerts": [dict(r) for r in rows]}


@app.post("/v1/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    conn = get_db()
    conn.execute("UPDATE alerts SET acknowledged=1 WHERE alert_id=?", (alert_id,))
    conn.commit()
    conn.close()
    return {"alert_id": alert_id, "acknowledged": True}


# ──────────────────────────────────────────────────────────────────────────────
# REST — Geofences
# ──────────────────────────────────────────────────────────────────────────────

class GeofenceCreate(BaseModel):
    name: str
    type: Optional[str] = "exclusion"
    coordinates: list[dict]
    altitude_min: Optional[float] = None
    altitude_max: Optional[float] = None


@app.get("/v1/geofences")
async def list_geofences():
    conn = get_db()
    rows = conn.execute("SELECT * FROM geofences WHERE active=1").fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["coordinates"] = json.loads(d["coordinates"])
        out.append(d)
    return {"geofences": out}


@app.post("/v1/geofences")
async def create_geofence(payload: GeofenceCreate):
    conn = get_db()
    ts = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO geofences (name, type, coordinates, created_at) VALUES (?,?,?,?)",
        (payload.name, payload.type, json.dumps([c if isinstance(c, dict) else list(c) for c in payload.coordinates]), ts),
    )
    conn.commit()
    gid = cur.lastrowid
    conn.close()
    return {"id": gid, "name": payload.name, "type": payload.type, "created_at": ts}


@app.delete("/v1/geofences/{geo_id}")
async def delete_geofence(geo_id: int):
    conn = get_db()
    conn.execute("UPDATE geofences SET active=0 WHERE id=?", (geo_id,))
    conn.commit()
    conn.close()
    return {"id": geo_id, "deleted": True}


# ──────────────────────────────────────────────────────────────────────────────
# REST — Missions
# ──────────────────────────────────────────────────────────────────────────────

class MissionCreate(BaseModel):
    mission_type: Optional[str] = "SURVEY"
    name: Optional[str] = None
    objectives: Optional[list[str]] = None
    priority: Optional[str] = "MEDIUM"
    pattern: Optional[str] = "grid"
    altitude_m: Optional[int] = 120
    area: Optional[list[dict]] = None
    asset_ids: Optional[list[str]] = None
    target_lat: Optional[float] = None
    target_lon: Optional[float] = None


@app.get("/v1/missions")
async def list_missions(limit: int = 50):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM missions ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["objectives"] = json.loads(d["objectives"])
        d["asset_ids"] = json.loads(d["asset_ids"] or "[]")
        out.append(d)
    return out


@app.post("/v1/missions")
async def create_mission(payload: MissionCreate):
    mission_id = f"MSN-{uuid.uuid4().hex[:8].upper()}"
    ts = datetime.now(timezone.utc).isoformat()
    name = payload.name or f"{payload.mission_type} {mission_id[-6:]}"
    objectives = payload.objectives or [f"{payload.mission_type} mission"]
    conn = get_db()
    conn.execute(
        """INSERT INTO missions
           (mission_id, name, mission_type, objectives, status, priority, pattern,
            altitude_m, area, asset_ids, created_at, started_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            mission_id, name, payload.mission_type, json.dumps(objectives),
            "ACTIVE", payload.priority, payload.pattern, payload.altitude_m,
            json.dumps(payload.area or []), json.dumps(payload.asset_ids or []),
            ts, ts,
        ),
    )
    conn.commit()
    conn.close()
    return {
        "mission_id": mission_id,
        "name": name,
        "status": "ACTIVE",
        "created_at": ts,
        "started_at": ts,
    }


@app.get("/v1/missions/{mission_id}/replay/timeline")
async def mission_replay_timeline(mission_id: str):
    return {
        "mission_id": mission_id,
        "count": 0,
        "start": None,
        "end": None,
        "timestamps": [],
    }


@app.get("/v1/missions/{mission_id}/replay/snapshot")
async def mission_replay_snapshot(mission_id: str, ts: Optional[str] = None, index: Optional[int] = None):
    return {"mission_id": mission_id, "entities": [], "ts": ts}


# ──────────────────────────────────────────────────────────────────────────────
# REST — Tasks / Dispatch
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/v1/tasks")
async def dispatch_task(payload: dict):
    task_id = f"TSK-{uuid.uuid4().hex[:8].upper()}"
    return {"task_id": task_id, "status": "QUEUED", "asset_id": payload.get("asset_id")}


@app.get("/v1/tasks/pending")
async def list_pending_tasks():
    return []


@app.post("/v1/tasks/{task_id}/approve")
async def approve_task(task_id: str, payload: dict = {}):
    return {"task_id": task_id, "status": "APPROVED", "approved_by": payload.get("approved_by", "operator")}


# ──────────────────────────────────────────────────────────────────────────────
# REST — Agent commands (HALT / RTB / camera)
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/agents")
async def agent_command(payload: dict):
    entity_id = payload.get("entity_id", "unknown")
    command = payload.get("command", "unknown")
    print(f"  [AGENT] {command.upper()} → {entity_id}")
    return {
        "entity_id": entity_id,
        "command": command,
        "status": "acknowledged",
        "ts": datetime.now(timezone.utc).isoformat(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# REST — Assets
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/v1/assets")
async def list_assets():
    assets = []
    for tmpl in ENTITY_TEMPLATES:
        if tmpl["domain"] in ("aerial", "ground") and tmpl["entity_type"] == "active":
            asset: dict[str, Any] = {
                "asset_id": tmpl["entity_id"],
                "type": tmpl["classification"],
                "capabilities": ["camera", "telemetry"],
                "link": "STRONG",
            }
            if tmpl["entity_id"] in _entity_batteries:
                asset["battery"] = round(_entity_batteries[tmpl["entity_id"]], 1)
            if tmpl.get("callsign"):
                asset["callsign"] = tmpl["callsign"]
            assets.append(asset)
    return {"assets": assets}


# ──────────────────────────────────────────────────────────────────────────────
# REST — Mission NLP parse
# ──────────────────────────────────────────────────────────────────────────────

_NLP_KEYWORDS = {
    "SURVEY":    ["survey", "scan", "map", "cover", "search", "look"],
    "MONITOR":   ["monitor", "watch", "observe", "track", "follow"],
    "PERIMETER": ["perimeter", "boundary", "border", "fence", "ring", "circle"],
    "ORBIT":     ["orbit", "loiter", "circle", "hover"],
    "DELIVER":   ["deliver", "drop", "bring", "carry", "supply"],
    "INSPECT":   ["inspect", "check", "examine", "assess"],
}

_PATTERN_KEYWORDS = {
    "grid":             ["grid", "lawnmower", "crosshatch"],
    "spiral":           ["spiral", "inward", "outward"],
    "expanding_square": ["expanding", "square", "creeping"],
    "orbit":            ["orbit", "circle", "loiter"],
    "perimeter":        ["perimeter", "boundary", "edge"],
}


def _parse_nlp(text: str) -> dict:
    lower = text.lower()

    mission_type = "SURVEY"
    for mt, words in _NLP_KEYWORDS.items():
        if any(w in lower for w in words):
            mission_type = mt
            break

    pattern_defaults = {
        "SURVEY": "grid", "MONITOR": "orbit", "SEARCH": "expanding_square",
        "PERIMETER": "perimeter", "ORBIT": "orbit", "DELIVER": "grid", "INSPECT": "grid",
    }
    pattern = pattern_defaults.get(mission_type, "grid")
    for pat, words in _PATTERN_KEYWORDS.items():
        if any(w in lower for w in words):
            pattern = pat
            break

    # Altitude — look for digits followed by "m" or "ft" or "meters"
    import re
    altitude_m = 120
    alt_match = re.search(r"(\d+)\s*(?:m\b|meters?|ft\b|feet)", lower)
    if alt_match:
        val = int(alt_match.group(1))
        if "ft" in alt_match.group(0) or "feet" in alt_match.group(0):
            val = int(val * 0.3048)
        altitude_m = max(20, min(500, val))

    # Asset hint — look for callsigns
    asset_hint = None
    for tmpl in ENTITY_TEMPLATES:
        cs = (tmpl.get("callsign") or "").lower()
        if cs and cs in lower:
            asset_hint = tmpl["callsign"]
            break

    confidence = 0.75 + (0.15 if asset_hint else 0) + (0.05 if alt_match else 0)
    interpretation = f"{mission_type} mission, {pattern} pattern at {altitude_m}m"
    if asset_hint:
        interpretation += f", using {asset_hint}"

    return {
        "mission_type": mission_type,
        "pattern": pattern,
        "altitude_m": altitude_m,
        "asset_hint": asset_hint,
        "objectives": [f"{mission_type} area at {altitude_m}m"],
        "confidence": round(min(confidence, 0.97), 2),
        "interpretation": interpretation,
    }


@app.post("/v1/missions/parse")
async def parse_mission_nlp(payload: dict):
    text = payload.get("text", "")
    if not text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    return _parse_nlp(text)


# ──────────────────────────────────────────────────────────────────────────────
# REST — Waypoint preview
# ──────────────────────────────────────────────────────────────────────────────

def _bounding_box(area: list[dict]) -> tuple[float, float, float, float]:
    lats = [p["lat"] for p in area]
    lons = [p["lon"] for p in area]
    return min(lats), max(lats), min(lons), max(lons)


def _grid_waypoints(area: list[dict], alt_m: int) -> list[dict]:
    min_lat, max_lat, min_lon, max_lon = _bounding_box(area)
    spacing = 0.0015  # ~165m
    waypoints = []
    row = 0
    lat = min_lat
    while lat <= max_lat:
        lons = [min_lon + i * spacing for i in range(int((max_lon - min_lon) / spacing) + 1)]
        if row % 2 == 1:
            lons = list(reversed(lons))
        for lon in lons:
            waypoints.append({"lat": round(lat, 6), "lon": round(lon, 6), "alt": alt_m})
        lat += spacing
        row += 1
    return waypoints


def _orbit_waypoints(area: list[dict], alt_m: int) -> list[dict]:
    min_lat, max_lat, min_lon, max_lon = _bounding_box(area)
    clat = (min_lat + max_lat) / 2
    clon = (min_lon + max_lon) / 2
    radius = max((max_lat - min_lat), (max_lon - min_lon)) / 2
    waypoints = []
    for i in range(24):
        angle = 2 * math.pi * i / 24
        waypoints.append({
            "lat": round(clat + radius * math.cos(angle), 6),
            "lon": round(clon + radius * math.sin(angle), 6),
            "alt": alt_m,
        })
    return waypoints


def _perimeter_waypoints(area: list[dict], alt_m: int) -> list[dict]:
    return [{"lat": p["lat"], "lon": p["lon"], "alt": alt_m} for p in area] + \
           [{"lat": area[0]["lat"], "lon": area[0]["lon"], "alt": alt_m}]


def _expanding_square_waypoints(area: list[dict], alt_m: int) -> list[dict]:
    min_lat, max_lat, min_lon, max_lon = _bounding_box(area)
    clat = (min_lat + max_lat) / 2
    clon = (min_lon + max_lon) / 2
    waypoints = []
    step = 0.0012
    for ring in range(12):
        d = step * (ring + 1)
        corners = [
            (clat - d, clon - d), (clat - d, clon + d),
            (clat + d, clon + d), (clat + d, clon - d),
        ]
        for lat, lon in corners:
            waypoints.append({"lat": round(lat, 6), "lon": round(lon, 6), "alt": alt_m})
    return waypoints


def _spiral_waypoints(area: list[dict], alt_m: int) -> list[dict]:
    min_lat, max_lat, min_lon, max_lon = _bounding_box(area)
    clat = (min_lat + max_lat) / 2
    clon = (min_lon + max_lon) / 2
    max_r = max((max_lat - min_lat), (max_lon - min_lon)) / 2
    waypoints = []
    steps = 60
    for i in range(steps):
        r = max_r * i / steps
        angle = i * 0.5
        waypoints.append({
            "lat": round(clat + r * math.cos(angle), 6),
            "lon": round(clon + r * math.sin(angle), 6),
            "alt": alt_m,
        })
    return waypoints


@app.post("/v1/missions/preview")
async def preview_waypoints(payload: dict):
    area = payload.get("area", [])
    pattern = payload.get("pattern", "grid")
    alt_m = int(payload.get("altitude_m", 120))

    if len(area) < 3:
        raise HTTPException(status_code=400, detail="area must have at least 3 points")

    generators = {
        "grid": _grid_waypoints,
        "orbit": _orbit_waypoints,
        "perimeter": _perimeter_waypoints,
        "expanding_square": _expanding_square_waypoints,
        "spiral": _spiral_waypoints,
    }
    gen = generators.get(pattern, _grid_waypoints)
    waypoints = gen(area, alt_m)

    return {"waypoints": waypoints, "pattern": pattern, "count": len(waypoints)}


# ──────────────────────────────────────────────────────────────────────────────
# REST — AI Reasoning
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/reasoning/{entity_id}")
async def get_reasoning(entity_id: str):
    """Return contextual AI reasoning for the given entity."""
    tmpl = next((t for t in ENTITY_TEMPLATES if t["entity_id"] == entity_id), None)
    if not tmpl:
        return {"thoughts": []}

    now = datetime.now(timezone.utc)
    ts = lambda offset_s: (now.timestamp() - offset_s)
    fmt = lambda epoch: datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%H:%M:%SZ")

    thoughts = []
    if tmpl["entity_type"] == "alert":
        thoughts = [
            {"ts": fmt(ts(9)), "msg": f"Anomalous velocity detected: {tmpl['speed_mps']:.1f} m/s exceeds baseline for airspace class", "confidence": 0.91},
            {"ts": fmt(ts(6)), "msg": "Cross-referencing against known flight corridors — no active plan on file", "confidence": 0.87},
            {"ts": fmt(ts(2)), "msg": "RF signature inconsistent with registered UAS database. Flagging for operator review.", "confidence": 0.84},
        ]
    elif entity_id in _entity_batteries and _entity_batteries[entity_id] < 25:
        bat = _entity_batteries[entity_id]
        thoughts = [
            {"ts": fmt(ts(8)), "msg": f"Battery at {bat:.0f}% — estimated {int(bat * 0.25):.0f} min flight time remaining", "confidence": 0.96},
            {"ts": fmt(ts(4)), "msg": "Evaluating RTB window. Current position within return-to-base range.", "confidence": 0.94},
            {"ts": fmt(ts(1)), "msg": "Recommend RTB initiation within 2 minutes to maintain safe reserve.", "confidence": 0.92},
        ]
    else:
        lat, lon = _entity_positions.get(entity_id, (0, 0))
        hdg = math.degrees(_entity_headings.get(entity_id, 0)) % 360
        thoughts = [
            {"ts": fmt(ts(12)), "msg": f"Tracking {tmpl['classification']} on nominal trajectory — no anomalies detected", "confidence": 0.97},
            {"ts": fmt(ts(5)), "msg": f"{tmpl['speed_mps']:.1f} m/s, heading {hdg:.0f}° — consistent with filed mission profile", "confidence": 0.95},
            {"ts": fmt(ts(1)), "msg": "Entity within authorized airspace. No geofence violations. Confidence nominal.", "confidence": 0.96},
        ]

    return {"entity_id": entity_id, "thoughts": thoughts}


# ──────────────────────────────────────────────────────────────────────────────
# REST — World state + health
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/v1/worldstate")
async def world_state():
    return {
        "entity_count": len(ENTITY_TEMPLATES),
        "active_missions": 2,
        "alert_count": 1,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "ts": datetime.now(timezone.utc).isoformat()}


# ──────────────────────────────────────────────────────────────────────────────
# REST — Video (stub — real impl needs ffmpeg + HLS)
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/v1/video/hls/{stream_id}/start")
async def start_hls(stream_id: str, payload: dict = {}):
    return {"stream_id": stream_id, "status": "started", "hls_url": f"/v1/video/hls/{stream_id}/stream.m3u8"}


@app.delete("/v1/video/hls/{stream_id}")
async def stop_hls(stream_id: str):
    return {"stream_id": stream_id, "status": "stopped"}


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="info",
    )
