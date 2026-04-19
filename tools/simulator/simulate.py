"""
Summit.OS Entity Simulator
==========================
Pushes synthetic moving entities to the Summit.OS fabric HTTP API.
Simulates drones, UGVs, towers, and sensors for demos / development.

Usage:
    python simulate.py

Environment:
    FABRIC_URL   — fabric service base URL (default: http://localhost:8001)
    SIM_COUNT    — number of entities to simulate (default: 25)
    SIM_AREA     — center of simulation area "lat,lon" (default: 34.05,-118.24 = LA)
    SIM_RADIUS   — simulation radius in degrees (default: 0.5)
    SIM_INTERVAL — seconds between position updates (default: 2)
"""

import asyncio
import math
import os
import random
import time
import uuid
from typing import Optional

try:
    import httpx
except ImportError:
    raise SystemExit("pip install httpx")

FABRIC_URL   = os.getenv("FABRIC_URL",   "http://localhost:8001")
SIM_COUNT    = int(os.getenv("SIM_COUNT",    "25"))
SIM_INTERVAL = float(os.getenv("SIM_INTERVAL", "2"))
_area        = os.getenv("SIM_AREA", "34.05,-118.24").split(",")
CENTER_LAT   = float(_area[0])
CENTER_LON   = float(_area[1])
SIM_RADIUS   = float(os.getenv("SIM_RADIUS", "0.5"))

# Entity type distribution
ENTITY_TEMPLATES = [
    {"asset_type": "DRONE",  "entity_type": "active",  "weight": 0.50, "speed": 0.003, "alt_range": (50,  200)},
    {"asset_type": "UGV",    "entity_type": "active",  "weight": 0.20, "speed": 0.001, "alt_range": (0,   5)},
    {"asset_type": "SENSOR", "entity_type": "active",  "weight": 0.15, "speed": 0.0,   "alt_range": (0,   10)},
    {"asset_type": "TOWER",  "entity_type": "active",  "weight": 0.10, "speed": 0.0,   "alt_range": (20,  50)},
    {"asset_type": "DRONE",  "entity_type": "unknown", "weight": 0.05, "speed": 0.004, "alt_range": (30,  150)},
]

CALLSIGN_PREFIXES = {
    "DRONE":  ["ECHO", "FALCON", "RAVEN", "HAWK", "SWIFT"],
    "UGV":    ["ROVER", "TITAN", "BEAR",  "WOLF"],
    "SENSOR": ["NODE",  "RELAY", "POINT"],
    "TOWER":  ["TOWER", "POST",  "BASE"],
}


def _pick_template() -> dict:
    r = random.random()
    cumulative = 0.0
    for t in ENTITY_TEMPLATES:
        cumulative += t["weight"]
        if r <= cumulative:
            return t
    return ENTITY_TEMPLATES[0]


def _make_entity(idx: int) -> dict:
    tmpl    = _pick_template()
    atype   = tmpl["asset_type"]
    prefix  = random.choice(CALLSIGN_PREFIXES.get(atype, ["UNIT"]))
    callsign = f"{prefix}-{idx:02d}"

    lat = CENTER_LAT + random.uniform(-SIM_RADIUS, SIM_RADIUS)
    lon = CENTER_LON + random.uniform(-SIM_RADIUS, SIM_RADIUS)
    alt = random.uniform(*tmpl["alt_range"])

    return {
        "entity_id":   f"sim-{atype.lower()}-{idx:03d}-{uuid.uuid4().hex[:6]}",
        "callsign":    callsign,
        "entity_type": tmpl["entity_type"],
        "asset_type":  atype,
        "lat":         lat,
        "lon":         lon,
        "alt":         alt,
        "heading":     random.uniform(0, 360),
        "speed":       tmpl["speed"],
        "stationary":  tmpl["speed"] == 0.0,
        "battery":     random.uniform(40, 100) if atype in ("DRONE", "UGV") else None,
    }


def _move_entity(e: dict) -> dict:
    if e["stationary"]:
        return e
    rad     = math.radians(e["heading"])
    e["lat"] = e["lat"] + math.cos(rad) * e["speed"]
    e["lon"] = e["lon"] + math.sin(rad) * e["speed"]
    # Bounce off area boundary
    if abs(e["lat"] - CENTER_LAT) > SIM_RADIUS:
        e["heading"] = (180 - e["heading"]) % 360
    if abs(e["lon"] - CENTER_LON) > SIM_RADIUS:
        e["heading"] = (360 - e["heading"]) % 360
    # Drift heading slightly
    e["heading"] = (e["heading"] + random.uniform(-5, 5)) % 360
    # Battery drain
    if e["battery"] is not None:
        e["battery"] = max(5, e["battery"] - random.uniform(0, 0.1))
    return e


async def _push_entity(client: httpx.AsyncClient, e: dict) -> None:
    payload = {
        "entity_id":   e["entity_id"],
        "type":        e["entity_type"],
        "callsign":    e["callsign"],
        "position":    {"lat": e["lat"], "lon": e["lon"], "alt": e["alt"]},
        "last_seen":   int(time.time()),
        "properties": {
            "asset_type": e["asset_type"],
            "heading":    round(e["heading"], 1),
            "simulated":  True,
            **({"battery": round(e["battery"], 1)} if e["battery"] is not None else {}),
        },
    }
    try:
        await client.post(
            f"{FABRIC_URL}/api/v1/entities",
            json=payload,
            timeout=5.0,
        )
    except Exception:
        pass  # fabric not yet available — keep retrying


async def main() -> None:
    entities = [_make_entity(i) for i in range(SIM_COUNT)]
    print(f"Summit.OS Simulator — {SIM_COUNT} entities @ {FABRIC_URL}")
    print(f"Area: {CENTER_LAT:.3f}, {CENTER_LON:.3f} ± {SIM_RADIUS}°")
    print(f"Update interval: {SIM_INTERVAL}s  |  Ctrl+C to stop\n")

    async with httpx.AsyncClient() as client:
        tick = 0
        while True:
            for e in entities:
                _move_entity(e)
                await _push_entity(client, e)

            tick += 1
            if tick % 10 == 0:
                alive  = sum(1 for e in entities if e["entity_type"] == "active")
                unk    = sum(1 for e in entities if e["entity_type"] == "unknown")
                low_bat = sum(1 for e in entities if e.get("battery") and e["battery"] < 20)
                print(f"[tick {tick:5d}]  active={alive}  unknown={unk}  low-battery={low_bat}")

            await asyncio.sleep(SIM_INTERVAL)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSimulator stopped.")
