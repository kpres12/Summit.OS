"""
Heli.OS Entity Simulator
==========================
Pushes synthetic moving entities to the Heli.OS fabric HTTP API.
Spreads entities across multiple US cities for realistic coverage.

Usage:
    python simulate.py

Environment:
    FABRIC_URL   — fabric service base URL (default: http://localhost:8001)
    SIM_COUNT    — total entities to simulate (default: 120)
    SIM_INTERVAL — seconds between position updates (default: 2)
    SIM_AREAS    — override spawn areas "lat,lon,radius:count|..." (optional)
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
SIM_COUNT    = int(os.getenv("SIM_COUNT",    "120"))
SIM_INTERVAL = float(os.getenv("SIM_INTERVAL", "2"))

# Default spawn areas: major US metros + a few incident-style clusters
DEFAULT_AREAS = [
    # (lat, lon, radius_deg, weight, label)
    (34.05, -118.24, 0.6,  0.18, "LOS ANGELES"),
    (40.71,  -74.00, 0.5,  0.15, "NEW YORK"),
    (41.88,  -87.63, 0.5,  0.12, "CHICAGO"),
    (29.76,  -95.37, 0.5,  0.10, "HOUSTON"),
    (33.45, -112.07, 0.4,  0.08, "PHOENIX"),
    (47.61, -122.33, 0.4,  0.08, "SEATTLE"),
    (25.77,  -80.19, 0.4,  0.07, "MIAMI"),
    (39.74, -104.99, 0.4,  0.07, "DENVER"),
    (37.77, -122.42, 0.4,  0.07, "SAN FRANCISCO"),
    (32.72,  -97.33, 0.4,  0.05, "DALLAS"),
    (44.98,  -93.27, 0.4,  0.03, "MINNEAPOLIS"),  # wildfire cluster
]

ENTITY_TEMPLATES = [
    {"asset_type": "DRONE",    "entity_type": "active",  "weight": 0.40, "speed": 0.004, "alt_range": (30,  250)},
    {"asset_type": "AIRCRAFT", "entity_type": "neutral", "weight": 0.15, "speed": 0.010, "alt_range": (1500, 12000)},
    {"asset_type": "UGV",      "entity_type": "active",  "weight": 0.15, "speed": 0.001, "alt_range": (0,   5)},
    {"asset_type": "SENSOR",   "entity_type": "active",  "weight": 0.10, "speed": 0.0,   "alt_range": (0,   10)},
    {"asset_type": "TOWER",    "entity_type": "active",  "weight": 0.08, "speed": 0.0,   "alt_range": (20,  80)},
    {"asset_type": "VESSEL",   "entity_type": "neutral", "weight": 0.05, "speed": 0.002, "alt_range": (0,   0)},
    {"asset_type": "DRONE",    "entity_type": "unknown", "weight": 0.05, "speed": 0.005, "alt_range": (30,  150)},
    {"asset_type": "DRONE",    "entity_type": "alert",   "weight": 0.02, "speed": 0.006, "alt_range": (50,  200)},
]

CALLSIGN_PREFIXES = {
    "DRONE":    ["ECHO", "FALCON", "RAVEN", "HAWK", "SWIFT", "TALON", "VIPER", "KESTREL"],
    "AIRCRAFT": ["N", "AA", "UAL", "DAL", "SWA", "FLT"],
    "UGV":      ["ROVER", "TITAN", "BEAR", "WOLF", "BISON"],
    "SENSOR":   ["NODE", "RELAY", "POINT", "GRID"],
    "TOWER":    ["TOWER", "POST", "BASE", "RELAY"],
    "VESSEL":   ["VESSEL", "COASTAL", "MARINE"],
}


def _pick_template() -> dict:
    r = random.random()
    cumulative = 0.0
    for t in ENTITY_TEMPLATES:
        cumulative += t["weight"]
        if r <= cumulative:
            return t
    return ENTITY_TEMPLATES[0]


def _pick_area() -> tuple:
    weights = [a[3] for a in DEFAULT_AREAS]
    total = sum(weights)
    r = random.random() * total
    cumulative = 0.0
    for area in DEFAULT_AREAS:
        cumulative += area[3]
        if r <= cumulative:
            return area
    return DEFAULT_AREAS[0]


def _make_entity(idx: int) -> dict:
    tmpl = _pick_template()
    area = _pick_area()
    center_lat, center_lon, radius = area[0], area[1], area[2]

    atype = tmpl["asset_type"]
    prefix = random.choice(CALLSIGN_PREFIXES.get(atype, ["UNIT"]))
    if atype == "AIRCRAFT":
        callsign = f"{prefix}{random.randint(100, 9999)}"
    else:
        callsign = f"{prefix}-{idx:02d}"

    lat = center_lat + random.uniform(-radius, radius)
    lon = center_lon + random.uniform(-radius, radius)
    alt = random.uniform(*tmpl["alt_range"])

    return {
        "entity_id":    f"sim-{atype.lower()}-{idx:03d}-{uuid.uuid4().hex[:6]}",
        "callsign":     callsign,
        "entity_type":  tmpl["entity_type"],
        "asset_type":   atype,
        "lat":          lat,
        "lon":          lon,
        "alt":          alt,
        "heading":      random.uniform(0, 360),
        "speed":        tmpl["speed"],
        "stationary":   tmpl["speed"] == 0.0,
        "battery":      random.uniform(30, 100) if atype in ("DRONE", "UGV") else None,
        "center_lat":   center_lat,
        "center_lon":   center_lon,
        "radius":       radius,
    }


def _move_entity(e: dict) -> dict:
    if e["stationary"]:
        return e
    rad = math.radians(e["heading"])
    e["lat"] = e["lat"] + math.cos(rad) * e["speed"]
    e["lon"] = e["lon"] + math.sin(rad) * e["speed"]
    # Bounce off area boundary
    if abs(e["lat"] - e["center_lat"]) > e["radius"]:
        e["heading"] = (180 - e["heading"]) % 360
    if abs(e["lon"] - e["center_lon"]) > e["radius"]:
        e["heading"] = (360 - e["heading"]) % 360
    # Drift heading
    e["heading"] = (e["heading"] + random.uniform(-5, 5)) % 360
    # Battery drain
    if e["battery"] is not None:
        e["battery"] = max(5, e["battery"] - random.uniform(0, 0.08))
    return e


async def _push_entity(client: httpx.AsyncClient, e: dict) -> None:
    payload = {
        "entity_id":   e["entity_id"],
        "type":        e["entity_type"],
        "callsign":    e["callsign"],
        "position":    {"lat": e["lat"], "lon": e["lon"], "alt": e["alt"]},
        "last_seen":   int(time.time()),
        "properties": {
            "asset_type":    e["asset_type"],
            "heading":       round(e["heading"], 1),
            "speed_ms":      round(e["speed"] * 111000 / SIM_INTERVAL, 1),  # approx m/s
            "simulated":     True,
            "controllable":  e["entity_type"] == "active",
            **({"battery": round(e["battery"], 1)} if e["battery"] is not None else {}),
        },
    }
    try:
        await client.post(f"{FABRIC_URL}/api/v1/entities", json=payload, timeout=5.0)
    except Exception:
        pass


async def main() -> None:
    entities = [_make_entity(i) for i in range(SIM_COUNT)]

    # Count by area
    area_counts: dict[str, int] = {}
    for e in entities:
        key = next((a[4] for a in DEFAULT_AREAS if a[0] == e["center_lat"]), "?")
        area_counts[key] = area_counts.get(key, 0) + 1

    print(f"Heli.OS Simulator — {SIM_COUNT} entities across {len(DEFAULT_AREAS)} areas @ {FABRIC_URL}")
    for area, count in sorted(area_counts.items(), key=lambda x: -x[1]):
        print(f"  {area:<20} {count} entities")
    print(f"\nUpdate interval: {SIM_INTERVAL}s  |  Ctrl+C to stop\n")

    async with httpx.AsyncClient() as client:
        tick = 0
        while True:
            for e in entities:
                _move_entity(e)
                await _push_entity(client, e)

            tick += 1
            if tick % 10 == 0:
                by_type: dict[str, int] = {}
                for e in entities:
                    by_type[e["asset_type"]] = by_type.get(e["asset_type"], 0) + 1
                low_bat = sum(1 for e in entities if e.get("battery") and e["battery"] < 20)
                type_str = "  ".join(f"{k}:{v}" for k, v in sorted(by_type.items()))
                print(f"[tick {tick:5d}]  {type_str}  low-bat:{low_bat}")

            await asyncio.sleep(SIM_INTERVAL)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSimulator stopped.")
