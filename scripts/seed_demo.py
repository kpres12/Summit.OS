#!/usr/bin/env python3
"""
Heli.OS Demo Seed
===================
Seeds the platform with realistic demo entities and triggers the full
autonomous pipeline so a fresh docker compose up shows a working system,
not an empty map.

What this does:
  1. Registers four UAV/sensor assets in Fabric (appear on the map)
  2. Pushes telemetry every 5 s so assets move (--live mode)
  3. Injects a smoke detection observation into Redis observations_stream
     → Intelligence scores it CRITICAL
     → MissionPlanner auto-dispatches a SURVEY mission to Tasking
  4. Injects a secondary SAR observation (missing hiker) → SEARCH mission
  5. Prints live advisory and mission counts so you can watch the pipeline

Usage:
  # One-shot seed (run once after docker compose up):
  python scripts/seed_demo.py

  # Live mode — assets move, new detections every 30 s:
  python scripts/seed_demo.py --live

  # Point at a non-default stack:
  python scripts/seed_demo.py --fabric http://myhost:8001 --redis redis://myhost:6379

Requirements:
  pip install httpx redis
"""

import argparse
import asyncio
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

try:
    import httpx
except ImportError:
    print("Missing dependency: pip install httpx")
    sys.exit(1)

try:
    import redis.asyncio as aioredis
except ImportError:
    print("Missing dependency: pip install redis")
    sys.exit(1)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_FABRIC_URL      = "http://localhost:8001"
DEFAULT_TASKING_URL     = "http://localhost:8004"
DEFAULT_INTELLIGENCE_URL = "http://localhost:8003"
DEFAULT_REDIS_URL       = "redis://localhost:6379"

# ── Demo geography — Sierra Nevada, CA ────────────────────────────────────────
REGION_CENTER = {"lat": 37.8651, "lon": -119.5383}  # Yosemite Valley area

# UAV and sensor assets registered in Fabric
ASSETS: List[Dict[str, Any]] = [
    {
        "device_id": "uav-alpha",
        "label": "UAV Alpha",
        "type": "QUADCOPTER",
        "lat": 37.872, "lon": -119.548, "alt": 120.0,
        "status": "READY",
        "sensors": {"battery_pct": 91, "camera": "thermal+rgb", "link_rssi": -62},
    },
    {
        "device_id": "uav-bravo",
        "label": "UAV Bravo",
        "type": "FIXED_WING",
        "lat": 37.858, "lon": -119.521, "alt": 300.0,
        "status": "READY",
        "sensors": {"battery_pct": 78, "camera": "multispectral", "link_rssi": -70},
    },
    {
        "device_id": "uav-charlie",
        "label": "UAV Charlie",
        "type": "QUADCOPTER",
        "lat": 37.845, "lon": -119.555, "alt": 95.0,
        "status": "READY",
        "sensors": {"battery_pct": 55, "camera": "rgb", "link_rssi": -74},
    },
    {
        "device_id": "tower-ridge-01",
        "label": "Ridge Camera 01",
        "type": "TOWER",
        "lat": 37.881, "lon": -119.509, "alt": 2340.0,
        "status": "ONLINE",
        "sensors": {"camera": "thermal", "fov_deg": 120, "temp_c": 18.4},
    },
    {
        "device_id": "tower-ridge-02",
        "label": "Ridge Camera 02",
        "type": "TOWER",
        "lat": 37.850, "lon": -119.530, "alt": 2180.0,
        "status": "ONLINE",
        "sensors": {"camera": "thermal+rgb", "fov_deg": 90, "temp_c": 17.9},
    },
]

# Observations that trigger the intelligence pipeline
DEMO_OBSERVATIONS: List[Dict[str, Any]] = [
    # Triggers CRITICAL → SURVEY mission
    {
        "class": "smoke plume",
        "confidence": 0.91,
        "lat": 37.862,
        "lon": -119.534,
        "source": "tower-ridge-01",
        "attributes": {"temperature_c": 340.0, "wind_direction": "SW", "plume_size_m2": 1200},
        "delay_s": 0,
    },
    # Triggers HIGH → SEARCH mission
    {
        "class": "missing hiker",
        "confidence": 0.82,
        "lat": 37.843,
        "lon": -119.562,
        "source": "uav-bravo",
        "attributes": {"last_seen_hours": 6, "trail": "Half Dome"},
        "delay_s": 3,
    },
    # Triggers HIGH → INSPECT mission
    {
        "class": "power line damage",
        "confidence": 0.78,
        "lat": 37.875,
        "lon": -119.518,
        "source": "uav-alpha",
        "attributes": {"line_segment": "tower-14-to-15"},
        "delay_s": 6,
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jitter(val: float, pct: float = 0.002) -> float:
    return val + random.uniform(-pct, pct)


def _orbit(center_lat: float, center_lon: float, radius_deg: float, angle_deg: float):
    rad = math.radians(angle_deg)
    return (
        center_lat + radius_deg * math.sin(rad),
        center_lon + radius_deg * math.cos(rad),
    )


async def _wait_for_service(url: str, name: str, timeout: int = 60) -> bool:
    print(f"  Waiting for {name} at {url}/health ", end="", flush=True)
    deadline = time.time() + timeout
    async with httpx.AsyncClient(timeout=3.0) as client:
        while time.time() < deadline:
            try:
                r = await client.get(f"{url}/health")
                if r.status_code == 200:
                    print(" ready")
                    return True
            except Exception:
                pass
            print(".", end="", flush=True)
            await asyncio.sleep(2)
    print(" TIMEOUT")
    return False


# ── Core operations ───────────────────────────────────────────────────────────

async def register_assets(fabric_url: str) -> None:
    """POST telemetry for each asset so they appear on the map."""
    print("\n[1/4] Registering assets in Fabric...")
    async with httpx.AsyncClient(timeout=10.0) as client:
        for asset in ASSETS:
            payload = {
                "device_id": asset["device_id"],
                "timestamp": _ts(),
                "location": {"lat": asset["lat"], "lon": asset["lon"], "alt": asset["alt"]},
                "sensors": {**asset["sensors"], "type": asset["type"], "label": asset["label"]},
                "status": asset["status"],
            }
            try:
                r = await client.post(f"{fabric_url}/telemetry", json=payload)
                status = "OK" if r.status_code == 200 else f"HTTP {r.status_code}"
            except Exception as e:
                status = f"ERR {e}"
            print(f"  {asset['device_id']:<20} {asset['type']:<14} {status}")


async def inject_observations(redis_url: str) -> None:
    """Push demo observations into observations_stream so Intelligence processes them."""
    print("\n[2/4] Injecting observations into pipeline...")
    rc = await aioredis.from_url(redis_url, decode_responses=True)

    # Ensure consumer groups exist (Intelligence and Fusion both read this stream)
    for group in ("intelligence", "fusion"):
        try:
            await rc.xgroup_create("observations_stream", group, id="$", mkstream=True)
        except Exception:
            pass  # group already exists

    for obs in DEMO_OBSERVATIONS:
        if obs["delay_s"]:
            await asyncio.sleep(obs["delay_s"])

        record = {
            "class":      obs["class"],
            "confidence": str(obs["confidence"]),
            "lat":        str(obs["lat"]),
            "lon":        str(obs["lon"]),
            "source":     obs["source"],
            "ts_iso":     _ts(),
            "payload":    json.dumps({
                "class":      obs["class"],
                "confidence": obs["confidence"],
                "lat":        obs["lat"],
                "lon":        obs["lon"],
                "source":     obs["source"],
                "ts_iso":     _ts(),
                "attributes": obs.get("attributes", {}),
            }),
        }
        msg_id = await rc.xadd("observations_stream", record)
        print(f"  {obs['class']:<30} conf={obs['confidence']:.2f}  → stream id {msg_id}")

    await rc.aclose()


async def post_alerts(fabric_url: str) -> None:
    """Create pre-seeded alerts so the alert queue isn't empty."""
    print("\n[3/4] Creating demo alerts...")
    import uuid

    alerts = [
        {
            "alert_id": f"demo-{uuid.uuid4().hex[:8]}",
            "timestamp": _ts(),
            "severity": "critical",
            "location": {"lat": 37.862, "lon": -119.534},
            "description": "Smoke plume detected on Ridge Cam 01 — conf 0.91. UAV dispatched.",
            "source": "tower-ridge-01",
        },
        {
            "alert_id": f"demo-{uuid.uuid4().hex[:8]}",
            "timestamp": _ts(),
            "severity": "high",
            "location": {"lat": 37.843, "lon": -119.562},
            "description": "Missing hiker reported near Half Dome trail. UAV search initiated.",
            "source": "uav-bravo",
        },
        {
            "alert_id": f"demo-{uuid.uuid4().hex[:8]}",
            "timestamp": _ts(),
            "severity": "medium",
            "location": {"lat": 37.875, "lon": -119.518},
            "description": "Possible power line damage detected. Inspection mission queued.",
            "source": "uav-alpha",
        },
    ]

    async with httpx.AsyncClient(timeout=10.0) as client:
        for alert in alerts:
            try:
                r = await client.post(f"{fabric_url}/alerts", json=alert)
                status = "OK" if r.status_code == 200 else f"HTTP {r.status_code}"
            except Exception as e:
                status = f"ERR {e}"
            print(f"  [{alert['severity'].upper():<8}] {alert['description'][:60]}  {status}")


async def verify_pipeline(intelligence_url: str, tasking_url: str) -> None:
    """Wait briefly then show advisory and mission counts."""
    print("\n[4/4] Verifying pipeline output (waiting 5 s for processing)...")
    await asyncio.sleep(5)

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.get(f"{intelligence_url}/advisories?limit=10")
            advisories = r.json() if r.status_code == 200 else []
            print(f"  Advisories generated : {len(advisories)}")
            for adv in advisories[:5]:
                print(f"    [{adv.get('risk_level','?')}] {adv.get('message','')[:70]}")
        except Exception as e:
            print(f"  Could not reach Intelligence: {e}")

        try:
            r = await client.get(f"{tasking_url}/api/v1/missions?limit=10")
            missions = r.json() if r.status_code == 200 else []
            print(f"  Missions auto-dispatched: {len(missions)}")
            for m in missions[:5]:
                mtype = m.get("mission_type") or m.get("type", "?")
                title = m.get("title", "")[:50]
                print(f"    {mtype:<12} {title}")
        except Exception as e:
            print(f"  Could not reach Tasking: {e}")


# ── Live mode ─────────────────────────────────────────────────────────────────

async def live_loop(fabric_url: str, redis_url: str, interval: int = 5) -> None:
    """Continuously update asset positions and inject detections periodically."""
    print(f"\nLive mode — updating assets every {interval}s. Ctrl-C to stop.\n")

    angles = {a["device_id"]: random.uniform(0, 360) for a in ASSETS}
    radii  = {a["device_id"]: random.uniform(0.003, 0.012) for a in ASSETS}
    tick = 0

    live_obs = [
        {"class": "active fire front", "confidence": 0.94, "lat": 37.860, "lon": -119.540,
         "source": "tower-ridge-01", "attributes": {"temp_c": 820, "spread_rate": "fast"}},
        {"class": "vessel in distress", "confidence": 0.87, "lat": 37.832, "lon": -119.501,
         "source": "uav-bravo", "attributes": {"vessel_type": "kayak", "signal": "PLB"}},
        {"class": "unauthorized uav", "confidence": 0.89, "lat": 37.878, "lon": -119.525,
         "source": "tower-ridge-02", "attributes": {"altitude_m": 85, "direction": "NE"}},
        {"class": "crop blight", "confidence": 0.76, "lat": 37.841, "lon": -119.545,
         "source": "uav-charlie", "attributes": {"area_ha": 3.2, "crop": "mixed"}},
    ]

    async with httpx.AsyncClient(timeout=5.0) as client:
        rc = await aioredis.from_url(redis_url, decode_responses=True)
        try:
            while True:
                # Move UAVs in slow orbits
                for asset in ASSETS:
                    did = asset["device_id"]
                    if asset["type"] == "TOWER":
                        # Towers don't move
                        lat, lon = asset["lat"], asset["lon"]
                    else:
                        angles[did] = (angles[did] + 1.5) % 360
                        lat, lon = _orbit(
                            REGION_CENTER["lat"], REGION_CENTER["lon"],
                            radii[did], angles[did],
                        )
                    payload = {
                        "device_id": did,
                        "timestamp": _ts(),
                        "location": {"lat": lat, "lon": lon, "alt": asset["alt"]},
                        "sensors": {**asset["sensors"]},
                        "status": asset["status"],
                    }
                    try:
                        await client.post(f"{fabric_url}/telemetry", json=payload)
                    except Exception:
                        pass

                # Every 30 s inject a new detection
                if tick % (30 // interval) == 0 and tick > 0:
                    obs_template = random.choice(live_obs)
                    obs = {**obs_template, "ts_iso": _ts(),
                           "lat": _jitter(obs_template["lat"]),
                           "lon": _jitter(obs_template["lon"])}
                    record = {
                        "class": obs["class"],
                        "confidence": str(obs["confidence"]),
                        "lat": str(obs["lat"]),
                        "lon": str(obs["lon"]),
                        "source": obs["source"],
                        "ts_iso": obs["ts_iso"],
                        "payload": json.dumps({
                            k: v for k, v in obs.items() if k != "delay_s"
                        }),
                    }
                    await rc.xadd("observations_stream", record)
                    print(f"  [{_ts()[11:19]}] Detection → {obs['class']} conf={obs['confidence']:.2f}")

                tick += 1
                await asyncio.sleep(interval)

        except asyncio.CancelledError:
            pass
        finally:
            await rc.aclose()


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    fabric_url       = args.fabric.rstrip("/")
    tasking_url      = args.tasking.rstrip("/")
    intelligence_url = args.intelligence.rstrip("/")
    redis_url        = args.redis

    print("=" * 60)
    print("  Heli.OS Demo Seed")
    print("=" * 60)
    print(f"  Fabric:      {fabric_url}")
    print(f"  Tasking:     {tasking_url}")
    print(f"  Intelligence:{intelligence_url}")
    print(f"  Redis:       {redis_url}")
    print()

    # Health checks
    if not args.skip_wait:
        ok = await _wait_for_service(fabric_url, "Fabric")
        if not ok:
            print("\nFabric not ready. Is docker compose up?")
            sys.exit(1)

    await register_assets(fabric_url)
    await inject_observations(redis_url)
    await post_alerts(fabric_url)
    await verify_pipeline(intelligence_url, tasking_url)

    print("\n" + "=" * 60)
    print("  Seed complete.")
    print()
    print("  Console  → http://localhost:3002")
    print("  API docs → http://localhost:8000/docs")
    print("  Grafana  → http://localhost:3001  (admin / admin)")
    print("=" * 60)

    if args.live:
        await live_loop(fabric_url, redis_url, interval=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heli.OS demo seed")
    parser.add_argument("--fabric",       default=DEFAULT_FABRIC_URL)
    parser.add_argument("--tasking",      default=DEFAULT_TASKING_URL)
    parser.add_argument("--intelligence", default=DEFAULT_INTELLIGENCE_URL)
    parser.add_argument("--redis",        default=DEFAULT_REDIS_URL)
    parser.add_argument("--live",         action="store_true",
                        help="Keep running — assets move, detections fire every 30 s")
    parser.add_argument("--skip-wait",    action="store_true",
                        help="Skip health-check wait (useful if services are already up)")
    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\nStopped.")
