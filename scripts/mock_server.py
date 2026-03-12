"""
Summit.OS Mock Data Server

Single-file server that gives the console live data with zero Docker/services.
Runs on port 8000 — exactly where the console expects the API gateway.

Provides:
  WS  ws://localhost:8000/ws        — entity stream (live OpenSky aircraft)
  GET /alerts                       — sample alert queue
  GET /missions                     — empty missions list
  GET /geofences                    — empty geofences list
  GET /entities                     — current entity snapshot
  GET /api/version                  — version header

Usage:
    pip install httpx websockets
    python scripts/mock_server.py
    python scripts/mock_server.py --bbox "32,-118,36,-114"   # LA area
"""
import argparse
import asyncio
import json
import logging
import math
import os
import random
import sqlite3
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

DB_PATH = os.getenv("MOCK_DB", "data/mock.db")

try:
    import httpx
except ImportError:
    print("ERROR: pip install httpx websockets"); exit(1)
try:
    import websockets
    from websockets.server import serve
except ImportError:
    print("ERROR: pip install websockets"); exit(1)

from http.server import BaseHTTPRequestHandler
import threading
import socketserver

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("mock")

# ── Config ────────────────────────────────────────────────────────────────────
OPENSKY_URL = "https://opensky-network.org/api/states/all"
POLL_INTERVAL = 15  # seconds between OpenSky refreshes

# ── SQLite persistence ────────────────────────────────────────────────────────

def _init_db():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS entities
                    (entity_id TEXT PRIMARY KEY, data TEXT, updated_at INTEGER)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS agents
                    (agent_id TEXT PRIMARY KEY, data TEXT, created_at INTEGER)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS adapters
                    (adapter_id TEXT PRIMARY KEY, data TEXT, created_at INTEGER)""")
    conn.commit()
    conn.close()

def _db_load_entities() -> Dict[str, dict]:
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("SELECT data FROM entities").fetchall()
        conn.close()
        return {json.loads(r[0])["entity_id"]: json.loads(r[0]) for r in rows}
    except Exception:
        return {}

def _db_save_entity(entity: dict):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT OR REPLACE INTO entities (entity_id, data, updated_at) VALUES (?,?,?)",
                     (entity["entity_id"], json.dumps(entity), int(time.time())))
        conn.commit()
        conn.close()
    except Exception:
        pass

def _db_load_agents() -> List[dict]:
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("SELECT data FROM agents ORDER BY created_at DESC LIMIT 100").fetchall()
        conn.close()
        return [json.loads(r[0]) for r in rows]
    except Exception:
        return []

def _db_save_agent(agent: dict):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT OR REPLACE INTO agents (agent_id, data, created_at) VALUES (?,?,?)",
                     (agent["agent_id"], json.dumps(agent), int(time.time())))
        conn.commit()
        conn.close()
    except Exception:
        pass

def _db_load_adapters() -> List[dict]:
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("SELECT data FROM adapters ORDER BY created_at DESC").fetchall()
        conn.close()
        return [json.loads(r[0]) for r in rows]
    except Exception:
        return []

def _db_save_adapter(adapter: dict):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT OR REPLACE INTO adapters (adapter_id, data, created_at) VALUES (?,?,?)",
                     (adapter["id"], json.dumps(adapter), int(time.time())))
        conn.commit()
        conn.close()
    except Exception:
        pass


# ── Shared state ──────────────────────────────────────────────────────────────
_entities: Dict[str, dict] = {}
_ws_clients: Set = set()
_loop: asyncio.AbstractEventLoop = None  # set in main()
_adapters: List[dict] = []
_agents: List[dict] = []
_geofences: List[dict] = []

# ── OpenSky fetch ─────────────────────────────────────────────────────────────

def _sv_to_entity(sv: list) -> Optional[dict]:
    """Convert OpenSky state vector to EntityData shape the console expects."""
    try:
        icao      = sv[0]
        callsign  = (sv[1] or "").strip() or icao.upper()
        lon, lat  = sv[5], sv[6]
        baro_alt  = sv[7] or 0.0
        on_ground = bool(sv[8])
        velocity  = sv[9] or 0.0
        heading   = sv[10] or 0.0
        vert_rate = sv[11] or 0.0
        if lat is None or lon is None:
            return None
        # entity_type maps to console color: active=green, alert=red, unknown=amber
        entity_type = "alert" if velocity > 260 else "active"
        return {
            # EntityData shape — matches useEntityStream.ts
            "entity_id":      f"adsb-{icao}",
            "entity_type":    entity_type,
            "domain":         "aerial",
            "classification": "aircraft",
            "callsign":       callsign,
            "position": {
                "lat":         lat,
                "lon":         lon,
                "alt":         baro_alt,
                "heading_deg": heading,
            },
            "speed_mps":      velocity,
            "confidence":     1.0,
            "last_seen":      int(time.time()),
            "source_sensors": ["opensky-adsb"],
            "track_state":    "confirmed",
            "battery_pct":    None,
            "mission_id":     None,
        }
    except Exception:
        return None


async def _broadcast_entity(entity: dict):
    """Push a single entity update to all connected WS clients."""
    if not _ws_clients:
        return
    msg = json.dumps({"type": "entity_update", "data": entity})
    await asyncio.gather(
        *[ws.send(msg) for ws in list(_ws_clients)],
        return_exceptions=True,
    )


async def poll_opensky(bbox: str):
    params = {}
    if bbox:
        p = [float(x) for x in bbox.split(",")]
        params = {"lamin": p[0], "lomin": p[1], "lamax": p[2], "lomax": p[3]}

    while True:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(OPENSKY_URL, params=params)
                r.raise_for_status()
                states = r.json().get("states") or []
                new_entities = {}
                for sv in states:
                    e = _sv_to_entity(sv)
                    if e:
                        new_entities[e["entity_id"]] = e
                _entities.clear()
                _entities.update(new_entities)
                log.info(f"OpenSky: {len(_entities)} aircraft loaded")

                # Broadcast all entities to connected WS clients
                if _ws_clients:
                    msg = json.dumps({"type": "entity_batch", "data": list(_entities.values())})
                    await asyncio.gather(
                        *[ws.send(msg) for ws in list(_ws_clients)],
                        return_exceptions=True,
                    )
        except Exception as e:
            log.warning(f"OpenSky fetch failed: {e}")

        await asyncio.sleep(POLL_INTERVAL)


# ── WebSocket server ──────────────────────────────────────────────────────────

async def ws_handler(ws):
    _ws_clients.add(ws)
    log.info(f"WS client connected ({len(_ws_clients)} total)")
    try:
        # Send current snapshot immediately on connect
        if _entities:
            await ws.send(json.dumps({
                "type": "entity_batch",
                "data": list(_entities.values()),
            }))
        # Keep alive + forward any pings
        async for msg in ws:
            pass  # client messages not used
    except Exception:
        pass
    finally:
        _ws_clients.discard(ws)
        log.info(f"WS client disconnected ({len(_ws_clients)} remaining)")


# ── HTTP REST server ──────────────────────────────────────────────────────────

SAMPLE_ALERTS = [
    {
        "alert_id":    "alert-001",
        "severity":    "WARNING",
        "description": "Aircraft UAL1361 exceeding typical cruise velocity",
        "source":      "adsb-a1b2c3",
        "ts_iso":      datetime.now(timezone.utc).isoformat(),
    },
    {
        "alert_id":    "alert-002",
        "severity":    "INFO",
        "description": "OpenSky data feed active — live aircraft tracking",
        "source":      "opensky",
        "ts_iso":      datetime.now(timezone.utc).isoformat(),
    },
]


class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress default HTTP logging

    def _send(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Summit-API-Version", "1")
        self.send_header("X-Summit-OS-Version", "0.3.0-dev")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type,Authorization,X-Org-ID")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]

        if path in ("/alerts", "/api/alerts", "/v1/alerts"):
            return self._send({"alerts": SAMPLE_ALERTS, "total": len(SAMPLE_ALERTS)})

        if path in ("/missions", "/api/missions", "/v1/missions"):
            return self._send({"missions": [], "total": 0})

        if path in ("/geofences", "/api/geofences", "/v1/geofences"):
            return self._send({"geofences": _geofences, "total": len(_geofences)})

        if path in ("/entities", "/api/entities", "/v1/entities"):
            return self._send({"entities": list(_entities.values()), "total": len(_entities)})

        if path in ("/api/version", "/version", "/v1/version"):
            return self._send({"api_version": "1", "os_version": "0.3.0-dev", "service": "mock"})

        if path in ("/worldstate", "/v1/worldstate"):
            return self._send({"entities": list(_entities.values()), "total": len(_entities)})

        if path in ("/health", "/api/health"):
            return self._send({"status": "ok", "entities": len(_entities), "ws_clients": len(_ws_clients)})

        if path in ("/adapters", "/api/adapters"):
            # Augment with entity-derived adapters if none registered
            live = list(_adapters)
            if not live:
                by_src: dict = {}
                for e in _entities.values():
                    src = (e.get("source_sensors") or ["unknown"])[0]
                    if src not in by_src:
                        by_src[src] = {
                            "id": src, "name": src.replace("-", " ").title(),
                            "protocol": "adsb" if "adsb" in src or "opensky" in src else "mavlink" if "mavlink" in src else "unknown",
                            "connection": src, "status": "online",
                            "entity_count": 0, "last_seen": int(time.time()),
                        }
                    by_src[src]["entity_count"] += 1
                live = list(by_src.values())
            return self._send({"adapters": live})

        if path in ("/agents", "/api/agents"):
            return self._send({"agents": _agents})

        if path.startswith("/reasoning/"):
            entity_id = path[len("/reasoning/"):]
            entity = _entities.get(entity_id)
            thoughts = _build_reasoning(entity, entity_id)
            return self._send({"entity_id": entity_id, "thoughts": thoughts})

        self._send({"error": "not found", "path": path}, 404)

    def do_POST(self):
        path = self.path.split("?")[0]
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"

        if path in ("/entity", "/api/entity"):
            try:
                entity = json.loads(body)
                eid = entity.get("entity_id")
                if eid:
                    _entities[eid] = entity
                    _db_save_entity(entity)
                    # Schedule broadcast to WS clients on the event loop
                    asyncio.run_coroutine_threadsafe(
                        _broadcast_entity(entity), _loop
                    )
                    return self._send({"ok": True, "entity_id": eid})
            except Exception as e:
                return self._send({"ok": False, "error": str(e)}, 400)

        if path in ("/adapters", "/api/adapters"):
            try:
                payload = json.loads(body)
                adapter = {
                    "id": f"adapter-{len(_adapters) + 1}",
                    "name": payload.get("name") or payload.get("protocol", "unknown"),
                    "protocol": payload.get("protocol", "unknown"),
                    "connection": payload.get("connection", ""),
                    "status": "online",
                    "entity_count": 0,
                    "last_seen": int(time.time()),
                }
                _adapters.append(adapter)
                _db_save_adapter(adapter)
                log.info(f"Adapter registered: {adapter['name']} ({adapter['protocol']})")
                return self._send({"ok": True, "adapter": adapter})
            except Exception as exc:
                return self._send({"ok": False, "error": str(exc)}, 400)

        if path in ("/agents", "/api/agents"):
            try:
                import uuid
                payload = json.loads(body)
                command = payload.get("command", "")
                entity_id = payload.get("entity_id")

                # OPA-style safety gate (mock enforcement)
                HALT_COMMANDS = {"HALT", "RETURN_TO_BASE", "KILL_SWITCH"}
                if command.upper() in HALT_COMMANDS:
                    # Check entity is known before allowing override commands
                    entity = _entities.get(entity_id or "")
                    if entity_id and not entity:
                        return self._send({"ok": False, "error": f"OPA DENY: entity '{entity_id}' not found in world model", "policy": "entity_existence"}, 403)
                    # RTB requires entity to be aerial or ground domain
                    if command.upper() == "RETURN_TO_BASE" and entity:
                        if entity.get("domain") not in ("aerial", "ground"):
                            return self._send({"ok": False, "error": f"OPA DENY: RTB not applicable to {entity.get('domain')} domain", "policy": "domain_capability"}, 403)
                    log.info(f"OPA ALLOW: {command} for entity {entity_id or 'n/a'}")

                agent = {
                    "agent_id": str(uuid.uuid4())[:8],
                    "mission_objective": payload.get("mission_objective", command or ""),
                    "entity_id": entity_id,
                    "command": command,
                    "status": "RUNNING",
                    "created_at": int(time.time()),
                }
                _agents.append(agent)
                _db_save_agent(agent)
                log.info(f"Agent started: {agent['agent_id']} — {agent.get('mission_objective', '')[:60]}")
                return self._send({"ok": True, "agent_id": agent["agent_id"], "status": "RUNNING"})
            except Exception as exc:
                return self._send({"ok": False, "error": str(exc)}, 400)

        if path in ("/geofences", "/api/geofences", "/v1/geofences"):
            try:
                import uuid
                payload = json.loads(body)
                geo = {
                    "geofence_id": str(uuid.uuid4())[:8],
                    "name": payload.get("name", "unnamed"),
                    "type": payload.get("type", "exclusion"),
                    "coordinates": payload.get("coordinates", []),
                    "created_at": int(time.time()),
                }
                _geofences.append(geo)
                log.info(f"Geofence created: {geo['name']} ({geo['type']}, {len(geo['coordinates'])} vertices)")
                return self._send({"ok": True, "geofence_id": geo["geofence_id"]})
            except Exception as exc:
                return self._send({"ok": False, "error": str(exc)}, 400)

        self._send({"ok": True, "message": "mock server"})


def _build_reasoning(entity: Optional[dict], entity_id: str) -> List[dict]:
    """Generate context-aware reasoning for an entity."""
    now = int(time.time())
    if not entity:
        return [{"ts": datetime.utcnow().strftime("%H:%M:%SZ"), "msg": f"No data for entity {entity_id}", "confidence": 0.5}]
    thoughts = []
    speed = entity.get("speed_mps", 0)
    etype = entity.get("entity_type", "unknown")
    domain = entity.get("domain", "unknown")
    batt = entity.get("battery_pct")
    pos = entity.get("position", {})
    lat = pos.get("lat", 0)
    lon = pos.get("lon", 0)
    if etype == "alert":
        thoughts.append({"ts": datetime.utcfromtimestamp(now - 8).strftime("%H:%M:%SZ"),
                         "msg": f"Anomalous velocity {speed:.1f} m/s exceeds domain baseline for {domain}", "confidence": 0.91})
        thoughts.append({"ts": datetime.utcfromtimestamp(now - 4).strftime("%H:%M:%SZ"),
                         "msg": "Cross-referencing against known corridors — no authorized match found", "confidence": 0.87})
        thoughts.append({"ts": datetime.utcfromtimestamp(now - 1).strftime("%H:%M:%SZ"),
                         "msg": "Flagging for operator review. Recommend visual verification before action.", "confidence": 0.84})
    elif batt is not None and batt < 25:
        thoughts.append({"ts": datetime.utcfromtimestamp(now - 6).strftime("%H:%M:%SZ"),
                         "msg": f"Battery critical at {batt:.0f}% — estimating < 5 min operational time", "confidence": 0.96})
        thoughts.append({"ts": datetime.utcfromtimestamp(now - 2).strftime("%H:%M:%SZ"),
                         "msg": "RTB evaluation: current position within return range. Initiating advisory.", "confidence": 0.93})
    else:
        thoughts.append({"ts": datetime.utcfromtimestamp(now - 10).strftime("%H:%M:%SZ"),
                         "msg": f"Tracking {entity.get('classification', 'entity')} on nominal trajectory at ({lat:.4f}, {lon:.4f})", "confidence": 0.97})
        thoughts.append({"ts": datetime.utcfromtimestamp(now - 3).strftime("%H:%M:%SZ"),
                         "msg": f"Speed {speed:.1f} m/s, heading {pos.get('heading_deg', 0):.0f}° — consistent with expected mission profile", "confidence": 0.95})
    return thoughts


def start_http(port: int):
    with socketserver.TCPServer(("", port), MockHandler) as httpd:
        httpd.allow_reuse_address = True
        log.info(f"HTTP mock server on :{port}")
        httpd.serve_forever()


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(bbox: str):
    global _loop, _entities, _adapters, _agents
    _loop = asyncio.get_event_loop()

    # Init SQLite and restore persisted state
    _init_db()
    _entities = _db_load_entities()
    _adapters = _db_load_adapters()
    _agents = _db_load_agents()
    log.info(f"Loaded {len(_entities)} entities, {len(_adapters)} adapters, {len(_agents)} agents from DB")

    # Start HTTP server in a background thread
    t = threading.Thread(target=start_http, args=(8000,), daemon=True)
    t.start()

    log.info("="*50)
    log.info("  Summit.OS Mock Server")
    log.info("="*50)
    log.info(f"  REST  → http://localhost:8000")
    log.info(f"  WS    → ws://localhost:8001/ws  (Next.js proxy)")
    log.info(f"  Area  → {bbox or 'global'}")
    log.info("="*50)
    log.info("  Open http://localhost:3000")
    log.info("="*50)

    # Start WebSocket server on 8001 (Next.js will proxy /ws → here)
    async with serve(ws_handler, "localhost", 8001):
        log.info("WebSocket server on :8001")
        await poll_opensky(bbox)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", default="", help="lat_min,lon_min,lat_max,lon_max")
    args = parser.parse_args()
    asyncio.run(main(args.bbox))
