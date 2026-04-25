"""
Heli.OS Mock Data Server

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
    conn.execute("""CREATE TABLE IF NOT EXISTS entity_trail
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     entity_id TEXT NOT NULL, lat REAL, lon REAL, ts INTEGER)""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trail ON entity_trail(entity_id, ts)")
    conn.execute("""CREATE TABLE IF NOT EXISTS missions
                    (mission_id TEXT PRIMARY KEY, data TEXT, created_at INTEGER)""")
    conn.commit()
    conn.close()

def _db_record_trail(entity_id: str, lat: float, lon: float):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO entity_trail (entity_id, lat, lon, ts) VALUES (?,?,?,?)",
                     (entity_id, lat, lon, int(time.time())))
        # keep last 150 points per entity
        conn.execute("""DELETE FROM entity_trail WHERE id IN (
            SELECT id FROM entity_trail WHERE entity_id=? ORDER BY ts ASC
            LIMIT MAX(0, (SELECT COUNT(*) FROM entity_trail WHERE entity_id=?) - 150)
        )""", (entity_id, entity_id))
        conn.commit()
        conn.close()
    except Exception:
        pass

def _db_get_trail(entity_id: str) -> list:
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT lat, lon FROM entity_trail WHERE entity_id=? ORDER BY ts DESC LIMIT 80",
            (entity_id,)
        ).fetchall()
        conn.close()
        return [{"lat": r[0], "lon": r[1]} for r in reversed(rows)]
    except Exception:
        return []

def _db_save_mission(mission: dict):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT OR REPLACE INTO missions (mission_id, data, created_at) VALUES (?,?,?)",
                     (mission["mission_id"], json.dumps(mission), int(time.time())))
        conn.commit()
        conn.close()
    except Exception:
        pass

def _db_list_missions() -> list:
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("SELECT data FROM missions ORDER BY created_at DESC LIMIT 50").fetchall()
        conn.close()
        return [json.loads(r[0]) for r in rows]
    except Exception:
        return []

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

# ── Satellite simulation ───────────────────────────────────────────────────────

_SATELLITE_CATALOG = [
    {"id": "ISS",        "name": "ISS (ZARYA)",      "type": "station",       "inc": 51.6, "alt_km": 408, "period_min": 92.6,  "raan": 0.0},
    {"id": "CAPELLA-1",  "name": "Capella-1 (SAR)",  "type": "sar",           "inc": 97.5, "alt_km": 525, "period_min": 95.4,  "raan": 30.0},
    {"id": "CAPELLA-2",  "name": "Capella-2 (SAR)",  "type": "sar",           "inc": 97.5, "alt_km": 525, "period_min": 95.4,  "raan": 60.0},
    {"id": "CAPELLA-4",  "name": "Capella-4 (SAR)",  "type": "sar",           "inc": 97.5, "alt_km": 525, "period_min": 95.4,  "raan": 150.0},
    {"id": "KH11-A",     "name": "USA-KH11 Alpha",   "type": "reconnaissance","inc": 97.7, "alt_km": 440, "period_min": 93.5,  "raan": 80.0},
    {"id": "KH11-B",     "name": "USA-KH11 Bravo",   "type": "reconnaissance","inc": 97.7, "alt_km": 440, "period_min": 93.5,  "raan": 200.0},
    {"id": "BARSM-1",    "name": "BARS-M No.1 (RU)", "type": "reconnaissance","inc": 97.9, "alt_km": 480, "period_min": 94.5,  "raan": 110.0},
    {"id": "GAOFEN-7",   "name": "Gaofen-7 (CN)",    "type": "optical",       "inc": 97.7, "alt_km": 506, "period_min": 94.9,  "raan": 260.0},
    {"id": "MAXAR-WV3",  "name": "WorldView-3",      "type": "optical",       "inc": 97.9, "alt_km": 617, "period_min": 97.0,  "raan": 320.0},
    {"id": "PLANET-1",   "name": "Planet Dove-1",    "type": "optical",       "inc": 97.4, "alt_km": 475, "period_min": 94.1,  "raan": 190.0},
    {"id": "STARLINK-1", "name": "Starlink Group 6", "type": "comms",         "inc": 53.0, "alt_km": 550, "period_min": 95.5,  "raan": 45.0},
]

def _compute_satellite_position(sat: dict) -> dict:
    """Simplified circular orbit position — not TLE-accurate but visually realistic."""
    now = time.time()
    period_s = sat["period_min"] * 60
    # Mean anomaly from arbitrary epoch
    M = (2 * math.pi * (now % period_s) / period_s)
    inc_r = math.radians(sat["inc"])
    raan_r = math.radians(sat["raan"])

    # Position in orbital plane
    x_orb = math.cos(M)
    y_orb = math.sin(M)

    # Rotate by inclination
    x_ec = x_orb
    y_ec = y_orb * math.cos(inc_r)
    z_ec = y_orb * math.sin(inc_r)

    # Rotate by RAAN (and Earth rotation)
    earth_rot = (2 * math.pi * (now % 86400) / 86400)
    angle = raan_r - earth_rot
    x_eq = x_ec * math.cos(angle) - y_ec * math.sin(angle)
    y_eq = x_ec * math.sin(angle) + y_ec * math.cos(angle)

    lat = math.degrees(math.asin(max(-1, min(1, z_ec))))
    lon = math.degrees(math.atan2(y_eq, x_eq))

    speed_kms = 7.8 - (sat["alt_km"] - 400) * 0.003  # ~7.8 km/s at 400km
    heading = math.degrees(math.atan2(y_orb * math.cos(inc_r), -math.sin(M))) % 360

    return {
        "entity_id": f"sat-{sat['id'].lower()}",
        "name": sat["name"],
        "sat_id": sat["id"],
        "sat_type": sat["type"],
        "position": {"lat": round(lat, 4), "lon": round(lon, 4), "alt": sat["alt_km"] * 1000},
        "speed_kms": round(speed_kms, 2),
        "heading_deg": round(heading, 1),
        "period_min": sat["period_min"],
        "ts": int(now),
    }

def _get_satellite_positions() -> list:
    return [_compute_satellite_position(s) for s in _SATELLITE_CATALOG]


# ── GPS Jamming simulation ─────────────────────────────────────────────────────

_GPS_JAM_ZONES = [
    {"id": "jam-1", "name": "Eastern Mediterranean",  "lat": 33.5,  "lon": 36.0,  "radius_km": 280, "intensity": 0.85, "source": "ELINT"},
    {"id": "jam-2", "name": "Black Sea Region",        "lat": 44.0,  "lon": 33.0,  "radius_km": 220, "intensity": 0.72, "source": "ELINT"},
    {"id": "jam-3", "name": "Persian Gulf / Iraq",     "lat": 31.0,  "lon": 47.0,  "radius_km": 180, "intensity": 0.91, "source": "ELINT"},
    {"id": "jam-4", "name": "GPS Degradation — Sinai", "lat": 30.0,  "lon": 33.5,  "radius_km": 120, "intensity": 0.65, "source": "ELINT"},
    {"id": "jam-5", "name": "Baltic Corridor",         "lat": 57.5,  "lon": 24.0,  "radius_km": 160, "intensity": 0.44, "source": "ELINT"},
]

def _get_gpsjam() -> list:
    # Add slight variation to intensity to show live feel
    zones = []
    for z in _GPS_JAM_ZONES:
        zone = dict(z)
        zone["intensity"] = round(min(1.0, max(0.1, z["intensity"] + random.gauss(0, 0.03))), 2)
        zone["ts"] = int(time.time())
        zones.append(zone)
    return zones


# ── Maritime simulation ────────────────────────────────────────────────────────

_MARITIME_VESSELS = [
    {"id": "mmsi-123456", "name": "GULF PIONEER",  "type": "tanker",     "lat": 26.5,  "lon": 56.8,  "heading": 180, "speed_kts": 12.0},
    {"id": "mmsi-234567", "name": "PACIFIC GLORY", "type": "tanker",     "lat": 25.2,  "lon": 58.2,  "heading": 220, "speed_kts": 9.5},
    {"id": "mmsi-345678", "name": "MSC AURORA",    "type": "cargo",      "lat": 24.8,  "lon": 55.4,  "heading": 90,  "speed_kts": 14.2},
    {"id": "mmsi-456789", "name": "HORMUZ STAR",   "type": "tanker",     "lat": 27.1,  "lon": 57.1,  "heading": 10,  "speed_kts": 0.0,  "status": "anchored"},
    {"id": "mmsi-567890", "name": "ENDEAVOUR",     "type": "container",  "lat": 22.3,  "lon": 59.8,  "heading": 290, "speed_kts": 16.0},
    {"id": "mmsi-678901", "name": "RED SEA HAWK",  "type": "tanker",     "lat": 14.5,  "lon": 42.8,  "heading": 330, "speed_kts": 11.0},
    {"id": "mmsi-789012", "name": "SUEZ CARRIER",  "type": "cargo",      "lat": 30.1,  "lon": 32.5,  "heading": 350, "speed_kts": 8.0},
]

_maritime_positions: Dict[str, dict] = {v["id"]: dict(v) for v in _MARITIME_VESSELS}

def _tick_maritime():
    for vid, v in _maritime_positions.items():
        if v.get("status") == "anchored" or v["speed_kts"] == 0:
            continue
        hdg_r = math.radians(v["heading"])
        d_deg = v["speed_kts"] * 0.000514 * 15 / 111000  # 15-second tick in degrees
        v["lat"] = round(v["lat"] + d_deg * math.cos(hdg_r), 5)
        v["lon"] = round(v["lon"] + d_deg * math.sin(hdg_r) / max(math.cos(math.radians(v["lat"])), 0.01), 5)
        v["heading"] = (v["heading"] + random.gauss(0, 0.5)) % 360

def _get_maritime() -> list:
    return list(_maritime_positions.values())


# ── No-fly zones ───────────────────────────────────────────────────────────────

_NO_FLY_ZONES = [
    {
        "id": "nfz-iran",
        "name": "Iran FIR Closure",
        "severity": "CRITICAL",
        "source": "NOTAM",
        "coordinates": [
            {"lat": 25.0, "lon": 44.0}, {"lat": 25.0, "lon": 63.5},
            {"lat": 39.5, "lon": 63.5}, {"lat": 39.5, "lon": 44.0},
        ],
        "active": True,
    },
    {
        "id": "nfz-iraq",
        "name": "Iraq LTMA Active",
        "severity": "HIGH",
        "source": "NOTAM",
        "coordinates": [
            {"lat": 29.0, "lon": 38.8}, {"lat": 29.0, "lon": 48.5},
            {"lat": 37.5, "lon": 48.5}, {"lat": 37.5, "lon": 38.8},
        ],
        "active": True,
    },
    {
        "id": "nfz-ukraine",
        "name": "Ukraine Conflict Zone",
        "severity": "CRITICAL",
        "source": "EUROCONTROL",
        "coordinates": [
            {"lat": 44.0, "lon": 22.0}, {"lat": 44.0, "lon": 40.2},
            {"lat": 52.5, "lon": 40.2}, {"lat": 52.5, "lon": 22.0},
        ],
        "active": True,
    },
]

def _get_noflyzones() -> list:
    return _NO_FLY_ZONES


# ── NLP mission parse ──────────────────────────────────────────────────────────

_NLP_MISSION_TYPES = {
    "SURVEY":    ["survey", "scan", "map", "search", "cover"],
    "MONITOR":   ["monitor", "watch", "observe", "track"],
    "PERIMETER": ["perimeter", "boundary", "border", "fence"],
    "ORBIT":     ["orbit", "loiter", "circle", "hover"],
    "DELIVER":   ["deliver", "drop", "bring", "supply"],
    "INSPECT":   ["inspect", "check", "examine", "assess"],
}
_NLP_PATTERNS = {
    "grid": ["grid", "lawnmower"], "spiral": ["spiral"],
    "expanding_square": ["expanding", "square"], "orbit": ["orbit", "circle"],
    "perimeter": ["perimeter", "boundary", "edge"],
}
_PATTERN_DEFAULTS = {
    "SURVEY": "grid", "MONITOR": "orbit", "SEARCH": "expanding_square",
    "PERIMETER": "perimeter", "ORBIT": "orbit", "DELIVER": "grid", "INSPECT": "grid",
}

def _parse_nlp(text: str) -> dict:
    import re
    lower = text.lower()
    mission_type = "SURVEY"
    for mt, words in _NLP_MISSION_TYPES.items():
        if any(w in lower for w in words):
            mission_type = mt
            break
    pattern = _PATTERN_DEFAULTS.get(mission_type, "grid")
    for pat, words in _NLP_PATTERNS.items():
        if any(w in lower for w in words):
            pattern = pat
            break
    altitude_m = 120
    alt_match = re.search(r"(\d+)\s*(?:m\b|meters?|ft\b|feet)", lower)
    if alt_match:
        val = int(alt_match.group(1))
        if "ft" in alt_match.group(0) or "feet" in alt_match.group(0):
            val = int(val * 0.3048)
        altitude_m = max(20, min(500, val))
    asset_hint = None
    for e in _entities.values():
        cs = (e.get("callsign") or "").lower()
        if cs and cs in lower:
            asset_hint = e.get("callsign")
            break
    confidence = round(min(0.97, 0.75 + (0.15 if asset_hint else 0) + (0.05 if alt_match else 0)), 2)
    return {
        "mission_type": mission_type, "pattern": pattern, "altitude_m": altitude_m,
        "asset_hint": asset_hint, "objectives": [f"{mission_type} area at {altitude_m}m"],
        "confidence": confidence,
        "interpretation": f"{mission_type} mission, {pattern} pattern at {altitude_m}m" + (f", using {asset_hint}" if asset_hint else ""),
    }


# ── Waypoint preview ───────────────────────────────────────────────────────────

def _preview_waypoints(area: list, pattern: str, alt_m: int) -> list:
    if not area or len(area) < 3:
        return []
    lats = [p["lat"] for p in area]; lons = [p["lon"] for p in area]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    clat = (min_lat + max_lat) / 2; clon = (min_lon + max_lon) / 2

    if pattern == "grid":
        # Scale spacing so we get ~10 rows and ~10 columns regardless of area size.
        # Minimum 0.002° (~220m at equator) to avoid absurd waypoint counts.
        sp_lat = max(0.002, (max_lat - min_lat) / 10)
        sp_lon = max(0.002, (max_lon - min_lon) / 10)
        wps, row = [], 0
        lat = min_lat
        while lat <= max_lat + 1e-9:
            cols = []
            lon = min_lon
            while lon <= max_lon + 1e-9:
                cols.append(lon)
                lon += sp_lon
            if row % 2:
                cols = list(reversed(cols))
            for lon in cols:
                wps.append({"lat": round(lat, 6), "lon": round(lon, 6), "alt": alt_m})
            lat += sp_lat; row += 1
        return wps
    elif pattern == "orbit":
        r = max(max_lat - min_lat, max_lon - min_lon) / 2
        return [{"lat": round(clat + r * math.cos(2*math.pi*i/24), 6),
                 "lon": round(clon + r * math.sin(2*math.pi*i/24), 6), "alt": alt_m} for i in range(24)]
    elif pattern == "perimeter":
        return [{"lat": p["lat"], "lon": p["lon"], "alt": alt_m} for p in area] + \
               [{"lat": area[0]["lat"], "lon": area[0]["lon"], "alt": alt_m}]
    elif pattern == "expanding_square":
        wps = []
        for ring in range(10):
            d = 0.0012 * (ring + 1)
            for lat, lon in [(clat-d,clon-d),(clat-d,clon+d),(clat+d,clon+d),(clat+d,clon-d)]:
                wps.append({"lat": round(lat,6), "lon": round(lon,6), "alt": alt_m})
        return wps
    else:  # spiral
        wps = []
        for i in range(50):
            r = max(max_lat-min_lat,max_lon-min_lon)/2 * i / 50
            a = i * 0.5
            wps.append({"lat": round(clat + r*math.cos(a),6), "lon": round(clon + r*math.sin(a),6), "alt": alt_m})
        return wps

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
                # Record trail points (sample every entity)
                for e in new_entities.values():
                    pos = e.get("position", {})
                    if pos.get("lat") and pos.get("lon"):
                        _db_record_trail(e["entity_id"], pos["lat"], pos["lon"])

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
        self.send_header("X-Heli-API-Version", "1")
        self.send_header("X-Heli-OS-Version", "0.3.0-dev")
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

        # ── Entity trail ────────────────────────────────────────────────────
        if "/entities/" in path and path.endswith("/trail"):
            parts = path.split("/")
            entity_id = parts[-2] if len(parts) >= 2 else ""
            trail = _db_get_trail(entity_id)
            if not trail and entity_id in _entities:
                pos = _entities[entity_id].get("position", {})
                if pos.get("lat") and pos.get("lon"):
                    trail = [{"lat": pos["lat"], "lon": pos["lon"]}]
            return self._send({"entity_id": entity_id, "trail": trail})

        # ── Assets list ─────────────────────────────────────────────────────
        if path in ("/v1/assets", "/assets"):
            assets = []
            for e in _entities.values():
                if e.get("domain") in ("aerial", "ground") and e.get("entity_type") == "active":
                    assets.append({
                        "asset_id": e["entity_id"],
                        "callsign": e.get("callsign", e["entity_id"][:8]),
                        "type": e.get("classification", "unknown"),
                        "capabilities": ["camera", "telemetry"],
                        "battery": e.get("battery_pct"),
                        "link": "STRONG",
                    })
            return self._send({"assets": assets[:20]})

        # ── Satellites ──────────────────────────────────────────────────────
        if path in ("/v1/satellites", "/satellites"):
            return self._send({"satellites": _get_satellite_positions()})

        # ── GPS Jamming ─────────────────────────────────────────────────────
        if path in ("/v1/gpsjam", "/gpsjam"):
            return self._send({"zones": _get_gpsjam(), "ts": int(time.time())})

        # ── Maritime ────────────────────────────────────────────────────────
        if path in ("/v1/maritime", "/maritime"):
            _tick_maritime()
            return self._send({"vessels": _get_maritime(), "ts": int(time.time())})

        # ── No-fly zones ────────────────────────────────────────────────────
        if path in ("/v1/noflyzones", "/noflyzones"):
            return self._send({"zones": _get_noflyzones()})

        # ── Task pending ────────────────────────────────────────────────────
        if path in ("/v1/tasks/pending",):
            return self._send([])

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

        # ── Mission create ──────────────────────────────────────────────────
        if path in ("/v1/missions", "/missions"):
            try:
                import uuid as _uuid
                payload = json.loads(body)
                mission_id = f"MSN-{_uuid.uuid4().hex[:8].upper()}"
                mission = {
                    "mission_id": mission_id,
                    "name": payload.get("name") or f"{payload.get('mission_type','SURVEY')} {mission_id[-6:]}",
                    "mission_type": payload.get("mission_type", "SURVEY"),
                    "objectives": payload.get("objectives", ["Mission objective"]),
                    "status": "ACTIVE",
                    "priority": payload.get("priority", "MEDIUM"),
                    "pattern": payload.get("pattern", "grid"),
                    "altitude_m": payload.get("altitude_m", 120),
                    "area": payload.get("area", []),
                    "asset_ids": payload.get("asset_ids", []),
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "started_at": datetime.utcnow().isoformat() + "Z",
                    "completed_at": None,
                }
                _db_save_mission(mission)
                log.info(f"Mission created: {mission_id} ({mission['mission_type']})")
                return self._send(mission)
            except Exception as exc:
                return self._send({"ok": False, "error": str(exc)}, 400)

        # ── Mission NLP parse ───────────────────────────────────────────────
        if path in ("/v1/missions/parse",):
            try:
                payload = json.loads(body)
                return self._send(_parse_nlp(payload.get("text", "")))
            except Exception as exc:
                return self._send({"ok": False, "error": str(exc)}, 400)

        # ── Mission waypoint preview ────────────────────────────────────────
        if path in ("/v1/missions/preview",):
            try:
                payload = json.loads(body)
                area = payload.get("area", [])
                pattern = payload.get("pattern", "grid")
                alt_m = int(payload.get("altitude_m", 120))
                wps = _preview_waypoints(area, pattern, alt_m)
                return self._send({"waypoints": wps, "pattern": pattern, "count": len(wps)})
            except Exception as exc:
                return self._send({"ok": False, "error": str(exc)}, 400)

        # ── Task dispatch ───────────────────────────────────────────────────
        if path in ("/v1/tasks",):
            try:
                import uuid as _uuid
                payload = json.loads(body)
                task_id = f"TSK-{_uuid.uuid4().hex[:8].upper()}"
                return self._send({"task_id": task_id, "status": "QUEUED", "asset_id": payload.get("asset_id")})
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


class _ReuseAddrTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def start_http(port: int):
    with _ReuseAddrTCPServer(("", port), MockHandler) as httpd:
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
    log.info("  Heli.OS Mock Server")
    log.info("="*50)
    log.info("  REST  → http://localhost:8000")
    log.info("  WS    → ws://localhost:8001/ws  (Next.js proxy)")
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
