"""
Summit.OS Tasking Simulation Executor (SITL/HITL scaffolding)

Purpose:
- Connect to one or more PX4/ArduPilot SITL endpoints via MAVLink
- Drive FireFlyAutopilot for simple demo flows (arm, takeoff, loiter plan)
- Optionally register assets in the Tasking Asset Registry

Usage examples:
  python apps/tasking/sim_executor.py \
    --asset drone-001=udp:127.0.0.1:14550 \
    --register-assets --arm --takeoff-alt 30 \
    --loiter-center 37.422,-122.084 --loiter-radius 150 --speed 5

Notes:
- Requires pymavlink for MAVLink integration
- Optionally requires httpx for asset registration calls
"""

import argparse
import asyncio
import os
import sys
from typing import Dict, Tuple, List
from dataclasses import dataclass

# Make local imports work when run as a script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

try:
    import httpx  # optional
    HTTPX_AVAILABLE = True
except Exception:
    HTTPX_AVAILABLE = False

from drone_autopilot import FireFlyAutopilot, Waypoint  # type: ignore


@dataclass
class SimAsset:
    asset_id: str
    conn: str


async def register_asset(tasking_url: str, asset: SimAsset, battery: float = 95.0, link: str = "OK") -> None:
    if not HTTPX_AVAILABLE:
        print("[sim] httpx not available, skipping asset registration")
        return
    try:
        payload = {
            "asset_id": asset.asset_id,
            "type": "sim",
            "capabilities": {"mavlink": True, "mavlink_conn": asset.conn},
            "battery": battery,
            "link": link,
            "constraints": {},
        }
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(f"{tasking_url}/api/v1/assets", json=payload)
            r.raise_for_status()
        print(f"[sim] registered asset {asset.asset_id} -> {tasking_url}")
    except Exception as e:
        print(f"[sim] failed to register asset {asset.asset_id}: {e}")


def parse_asset_arg(value: str) -> SimAsset:
    # Format: <asset_id>=<connection_string>
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected format <asset_id>=<connection_string>")
    aid, conn = value.split("=", 1)
    return SimAsset(asset_id=aid.strip(), conn=conn.strip())


async def simple_loiter_plan(center: Tuple[float, float], radius_m: float, altitude: float, speed: float) -> List[Waypoint]:
    lat, lon = center
    # single waypoint at offset radius to induce loiter
    return [Waypoint(lat=lat, lon=lon, alt=altitude, speed=speed, action="WAYPOINT")]  # type: ignore


async def run_sim(args: argparse.Namespace) -> None:
    # Build asset list
    assets: List[SimAsset] = args.assets
    if not assets:
        print("[sim] no assets provided. Use --asset drone-001=udp:127.0.0.1:14550")
        return

    # Connect all autopilots
    pilots: Dict[str, FireFlyAutopilot] = {}
    for a in assets:
        p = FireFlyAutopilot(device_id=a.asset_id, connection_string=a.conn)
        ok = await p.connect()
        if not ok:
            print(f"[sim] failed to connect {a.asset_id} at {a.conn}")
            continue
        pilots[a.asset_id] = p
        print(f"[sim] connected {a.asset_id} at {a.conn}")

    if not pilots:
        print("[sim] no connected assets, exiting")
        return

    # Optionally register assets with tasking service
    if args.register_assets:
        tasking_url = args.tasking_url
        await asyncio.gather(*[register_asset(tasking_url, a) for a in assets])

    # Optional arm + takeoff
    if args.arm:
        for a, p in pilots.items():
            ok = await p.arm()
            print(f"[sim] arm {a}: {ok}")
            if ok and args.takeoff_alt:
                ok2 = await p.takeoff(args.takeoff_alt)
                print(f"[sim] takeoff {a} to {args.takeoff_alt}m: {ok2}")

    # Optional loiter plan
    if args.loiter_center:
        lat, lon = args.loiter_center
        for a, p in pilots.items():
            wps = await simple_loiter_plan((lat, lon), args.loiter_radius, args.altitude, args.speed)
            ok = await p.set_mission(wps)
            print(f"[sim] set mission for {a}: {ok}")
            if ok and args.start:
                ok2 = await p.start_mission()
                print(f"[sim] start mission for {a}: {ok2}")

    # Monitor until interrupted
    print("[sim] running. Ctrl+C to stop.")
    try:
        while True:
            await asyncio.sleep(2.0)
            for a, p in pilots.items():
                st = p.get_status()
                prog = p.get_mission_progress()
                print(f"[sim] {a} mode={st.mode.value} armed={st.armed} batt={st.battery}% pos={st.position} prog={prog['state']}")
    except KeyboardInterrupt:
        print("[sim] shutting down...")
    finally:
        await asyncio.gather(*[p.disconnect() for p in pilots.values()])
        print("[sim] disconnected all")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summit.OS SITL/HITL simulation executor")
    p.add_argument(
        "--asset",
        dest="assets",
        action="append",
        type=parse_asset_arg,
        help="Register an asset mapping asset_id=connection_string (repeatable)",
        default=[],
    )
    p.add_argument(
        "--register-assets",
        action="store_true",
        help="Register assets in Tasking Asset Registry",
    )
    p.add_argument(
        "--tasking-url",
        type=str,
        default=os.getenv("TASKING_URL", "http://localhost:8004"),
        help="Tasking service base URL",
    )
    p.add_argument("--arm", action="store_true", help="Arm vehicles")
    p.add_argument("--takeoff-alt", type=float, default=0.0, help="If >0, takeoff to this altitude (m)")
    p.add_argument("--start", action="store_true", help="Start mission after upload")
    p.add_argument("--altitude", type=float, default=60.0, help="Default mission altitude (m)")
    p.add_argument("--speed", type=float, default=5.0, help="Default mission speed (m/s)")
    p.add_argument(
        "--loiter-center",
        type=lambda s: tuple(map(float, s.split(","))),
        help="Center lat,lon for loiter pattern",
    )
    p.add_argument("--loiter-radius", type=float, default=100.0, help="Loiter radius (m)")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run_sim(args))


if __name__ == "__main__":
    main()
