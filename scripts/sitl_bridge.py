"""
Summit.OS SITL Bridge

Starts a virtual ArduPilot drone at your coordinates, reads its MAVLink
telemetry, and pushes it into the mock server as a live entity on the map.

Usage:
    python scripts/sitl_bridge.py
    python scripts/sitl_bridge.py --lat 34.05 --lon -118.25 --name "Ghost-1"
    python scripts/sitl_bridge.py --count 3   # spawn 3 virtual drones

Requirements:
    .venv/bin/python scripts/sitl_bridge.py
    (dronekit-sitl + pymavlink installed in .venv)
"""
import argparse
import asyncio
import json
import math
import sys
import time

try:
    import dronekit_sitl
    import httpx
    from pymavlink import mavutil
except ImportError:
    print("ERROR: run with .venv/bin/python scripts/sitl_bridge.py")
    sys.exit(1)

MOCK_URL = "http://localhost:8000/entity"
TELEMETRY_HZ = 2   # updates per second


def battery_state(pct: float) -> str:
    if pct < 15:  return "alert"
    if pct < 25:  return "unknown"   # amber — low
    return "active"


def build_entity(vehicle_id: str, name: str, lat: float, lon: float,
                 alt: float, heading: float, airspeed: float,
                 groundspeed: float, battery_pct: float,
                 armed: bool, mode: str) -> dict:
    return {
        "entity_id":      vehicle_id,
        "entity_type":    battery_state(battery_pct),
        "domain":         "aerial",
        "classification": "drone",
        "callsign":       name,
        "position": {
            "lat":         lat,
            "lon":         lon,
            "alt":         alt,
            "heading_deg": heading,
        },
        "speed_mps":      groundspeed,
        "confidence":     1.0,
        "last_seen":      int(time.time()),
        "source_sensors": ["mavlink-sitl"],
        "track_state":    "confirmed",
        "battery_pct":    battery_pct,
        "mission_id":     None,
        # Extra fields for detail panel
        "armed":          armed,
        "flight_mode":    mode,
        "airspeed_mps":   airspeed,
    }


def run_drone(vehicle_id: str, name: str, lat: float, lon: float,
              connection_string: str):
    """Connect to SITL via MAVLink and stream telemetry to mock server."""
    print(f"[{name}] connecting to {connection_string} ...")
    conn = mavutil.mavlink_connection(connection_string, source_system=255)
    conn.wait_heartbeat(timeout=15)
    print(f"[{name}] MAVLink heartbeat received — drone online")

    # Request data streams
    conn.mav.request_data_stream_send(
        conn.target_system,
        conn.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL,
        4,   # 4 Hz
        1,   # start
    )

    # State tracking
    cur_lat = lat
    cur_lon = lon
    cur_alt = 0.0
    cur_heading = 0.0
    cur_airspeed = 0.0
    cur_groundspeed = 0.0
    cur_battery = 100.0
    cur_armed = False
    cur_mode = "STABILIZE"
    last_push = 0.0

    with httpx.Client(timeout=3.0) as client:
        while True:
            msg = conn.recv_match(blocking=True, timeout=1.0)
            if msg is None:
                continue

            t = msg.get_type()

            if t == "GLOBAL_POSITION_INT":
                cur_lat = msg.lat / 1e7
                cur_lon = msg.lon / 1e7
                cur_alt = msg.relative_alt / 1000.0
                cur_heading = msg.hdg / 100.0 if msg.hdg != 65535 else cur_heading

            elif t == "VFR_HUD":
                cur_airspeed    = msg.airspeed
                cur_groundspeed = msg.groundspeed
                cur_alt         = msg.alt

            elif t == "SYS_STATUS":
                v = msg.voltage_battery / 1000.0
                # Estimate battery % from voltage (3S LiPo: 12.6V full, 10.5V empty)
                cur_battery = max(0.0, min(100.0, (v - 10.5) / (12.6 - 10.5) * 100))

            elif t == "HEARTBEAT":
                cur_armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                try:
                    cur_mode = mavutil.mode_string_v10(msg)
                except Exception:
                    pass

            # Push at TELEMETRY_HZ
            now = time.time()
            if now - last_push >= 1.0 / TELEMETRY_HZ:
                entity = build_entity(
                    vehicle_id, name,
                    cur_lat, cur_lon, cur_alt, cur_heading,
                    cur_airspeed, cur_groundspeed, cur_battery,
                    cur_armed, cur_mode,
                )
                try:
                    client.post(MOCK_URL, json=entity)
                except Exception:
                    pass  # mock server not running — keep going
                last_push = now

                status = "ARMED" if cur_armed else "DISARMED"
                print(f"\r[{name}] {status} | {cur_mode} | "
                      f"alt={cur_alt:.1f}m spd={cur_groundspeed:.1f}m/s "
                      f"batt={cur_battery:.0f}% "
                      f"@{cur_lat:.4f},{cur_lon:.4f}    ",
                      end="", flush=True)


def launch_drone(index: int, lat: float, lon: float, name: str) -> tuple:
    """Start a SITL instance. Returns (sitl, connection_string)."""
    # Offset multiple drones slightly so they don't stack
    offset_lat = lat + (index * 0.0005)
    offset_lon = lon + (index * 0.0005)

    print(f"Starting SITL for {name} at {offset_lat:.4f},{offset_lon:.4f} ...")
    sitl = dronekit_sitl.start_default(lat=offset_lat, lon=offset_lon)
    print(f"  {name} SITL ready → {sitl.connection_string()}")
    return sitl, sitl.connection_string()


def main():
    parser = argparse.ArgumentParser(description="Summit.OS SITL Bridge")
    parser.add_argument("--lat",   type=float, default=34.0522, help="Home latitude")
    parser.add_argument("--lon",   type=float, default=-118.2437, help="Home longitude")
    parser.add_argument("--name",  default="Ghost-1", help="Drone name")
    parser.add_argument("--count", type=int, default=1, help="Number of virtual drones (max 3)")
    args = parser.parse_args()

    count = min(args.count, 3)
    names = [args.name] if count == 1 else [f"Ghost-{i+1}" for i in range(count)]
    vehicle_ids = [f"sitl-drone-{i+1}" for i in range(count)]

    print("=" * 50)
    print("  Summit.OS SITL Bridge")
    print("=" * 50)
    print(f"  Spawning {count} virtual drone(s) over {args.lat:.4f},{args.lon:.4f}")
    print(f"  Telemetry → {MOCK_URL}")
    print("=" * 50)

    # Launch all SITL instances
    instances = []
    for i in range(count):
        sitl, conn_str = launch_drone(i, args.lat, args.lon, names[i])
        instances.append((vehicle_ids[i], names[i], conn_str, sitl))

    if count == 1:
        # Single drone — run in main thread
        vid, name, conn_str, _ = instances[0]
        try:
            run_drone(vid, name, args.lat, args.lon, conn_str)
        except KeyboardInterrupt:
            print("\nShutting down...")
    else:
        # Multiple drones — threads
        import threading
        threads = []
        for vid, name, conn_str, _ in instances:
            t = threading.Thread(
                target=run_drone,
                args=(vid, name, args.lat, args.lon, conn_str),
                daemon=True,
            )
            t.start()
            threads.append(t)
            time.sleep(2)  # stagger startup
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")

    for _, _, _, sitl in instances:
        sitl.stop()


if __name__ == "__main__":
    main()
