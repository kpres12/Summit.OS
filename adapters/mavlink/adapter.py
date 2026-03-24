"""
Summit.OS MAVLink Adapter

Connects to MAVLink-compatible autopilots (ArduPilot, PX4) and publishes
real-time telemetry as Summit.OS TRACK entities into the data fabric.

Vehicle Config (MAVLINK_VEHICLES env var, path to JSON file):
    [
      {
        "vehicle_id": "drone-01",
        "connection_string": "udp:192.168.1.100:14550",
        "system_id": 1,
        "name": "Recon Alpha",
        "class_label": "quadcopter"
      }
    ]

Environment variables:
    MAVLINK_ENABLED         - "true" to enable (default: "false")
    MAVLINK_VEHICLES        - path to JSON vehicle config file
    MAVLINK_CONNECTION      - single vehicle connection string (default: "udp:127.0.0.1:14550")
    MAVLINK_VEHICLE_ID      - single vehicle ID (default: "drone-01")
    MAVLINK_TELEMETRY_HZ    - publish rate in Hz (default: 2, max: 10)
    MAVLINK_ORG_ID          - org_id (default: "")
    MQTT_HOST / MQTT_PORT   - broker connection
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))

from sdk import BaseAdapter, AdapterManifest, EntityBuilder, Protocol, Capability

logger = logging.getLogger("summit.adapter.mavlink")


def _load_vehicle_config(path: Optional[str]) -> List[Dict[str, Any]]:
    if path:
        try:
            with open(path) as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} vehicle definitions from {path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load vehicle config from {path}: {e}")
    return [{
        "vehicle_id": os.getenv("MAVLINK_VEHICLE_ID", "drone-01"),
        "connection_string": os.getenv("MAVLINK_CONNECTION", "udp:127.0.0.1:14550"),
        "system_id": 1,
        "name": os.getenv("MAVLINK_VEHICLE_ID", "drone-01"),
        "class_label": "uav",
    }]


class MAVLinkAdapter(BaseAdapter):
    """Manages MAVLink vehicle connections and publishes telemetry as TRACK entities."""

    MANIFEST = AdapterManifest(
        name="mavlink",
        version="1.0.0",
        protocol=Protocol.MAVLINK,
        capabilities=[Capability.READ, Capability.WRITE, Capability.STREAM],
        entity_types=["TRACK"],
        description="MAVLink drone telemetry adapter — ArduPilot, PX4, DJI",
        optional_env=["MAVLINK_VEHICLES", "MAVLINK_CONNECTION", "MAVLINK_TELEMETRY_HZ"],
    )

    def __init__(
        self,
        vehicles_path: Optional[str] = os.getenv("MAVLINK_VEHICLES"),
        telemetry_hz: float = min(float(os.getenv("MAVLINK_TELEMETRY_HZ", "2")), 10.0),
        org_id: str = os.getenv("MAVLINK_ORG_ID", ""),
        **kwargs,
    ):
        super().__init__(device_id="mavlink", org_id=org_id, **kwargs)
        self.publish_interval = 1.0 / max(telemetry_hz, 0.1)
        self.vehicles = _load_vehicle_config(vehicles_path)

    @property
    def enabled(self) -> bool:
        return os.getenv("MAVLINK_ENABLED", "false").lower() == "true"

    async def run(self):
        logger.info(
            f"MAVLink adapter running ({len(self.vehicles)} vehicle(s), "
            f"{1.0/self.publish_interval:.1f}Hz)"
        )
        tasks = [
            asyncio.create_task(self._run_vehicle(v))
            for v in self.vehicles
        ]
        while not self.stopped:
            await self.sleep(1.0)
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_vehicle(self, vehicle: Dict[str, Any]):
        vehicle_id = vehicle["vehicle_id"]
        connection_string = vehicle.get("connection_string", "udp:127.0.0.1:14550")

        try:
            packages_path = os.path.join(os.path.dirname(__file__), "..", "..", "packages")
            if packages_path not in sys.path:
                sys.path.insert(0, packages_path)
            from hal.drivers.mavlink import MAVLinkDriver

            driver = MAVLinkDriver(
                connection_string=connection_string,
                system_id=vehicle.get("system_id", 1),
            )
            connected = await driver.connect()
            if not connected:
                logger.error(f"MAVLink: could not connect to {vehicle_id} at {connection_string}")
                return
            await driver.start_telemetry()
            logger.info(f"MAVLink {vehicle_id}: telemetry streaming")

            # Subscribe to dispatch commands for this vehicle
            cmd_task = asyncio.create_task(
                self._command_listener(f"mavlink-{vehicle_id}", driver)
            )

            while not self.stopped:
                entity = self._telemetry_to_entity(driver.telemetry, vehicle)
                self.publish(entity, qos=0)
                await self.sleep(self.publish_interval)

            cmd_task.cancel()
            await asyncio.gather(cmd_task, return_exceptions=True)
            await driver.disconnect()

        except ImportError:
            logger.warning(f"MAVLink driver unavailable for {vehicle_id} — simulating")
            await self._run_simulated_vehicle(vehicle)
        except Exception as e:
            logger.error(f"MAVLink worker error for {vehicle_id}: {e}")

    async def _command_listener(self, entity_id: str, driver: Any):
        """Subscribe to tasks/{entity_id}/dispatch and execute commands on the driver."""
        import queue as _queue
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            logger.warning("paho-mqtt not available — command listener disabled")
            return

        cmd_queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _on_message(_client, _userdata, msg):
            try:
                payload = json.loads(msg.payload.decode("utf-8"))
            except Exception:
                return
            loop.call_soon_threadsafe(cmd_queue.put_nowait, payload)

        client = mqtt.Client(client_id=f"summit-mavlink-cmd-{entity_id}")
        if self._mqtt_username and self._mqtt_password:
            client.username_pw_set(self._mqtt_username, self._mqtt_password)
        client.on_message = _on_message

        try:
            client.connect(self._mqtt_host, self._mqtt_port, 60)
            client.loop_start()
            client.subscribe(f"tasks/{entity_id}/dispatch", qos=1)
            logger.info(f"MAVLink command listener active on tasks/{entity_id}/dispatch")
        except Exception as e:
            logger.error(f"Command listener MQTT connect failed for {entity_id}: {e}")
            return

        try:
            while True:
                payload = await cmd_queue.get()
                await self._execute_command(driver, entity_id, payload)
        except asyncio.CancelledError:
            pass
        finally:
            client.loop_stop()
            client.disconnect()

    async def _execute_command(self, driver: Any, entity_id: str, payload: Dict[str, Any]):
        """Execute a dispatched command on the MAVLink driver."""
        action = (payload.get("action") or "").upper()
        waypoints = payload.get("waypoints") or []

        logger.info(f"MAVLink {entity_id}: executing action={action} waypoints={len(waypoints)}")

        if action == "HALT":
            await driver.land()
        elif action in ("RTL", "RTB"):
            await driver.rtl()
        elif action == "TAKEOFF":
            alt = payload.get("altitude", 10.0)
            await driver.takeoff(altitude=float(alt))
        elif waypoints:
            # Fly to first waypoint; subsequent waypoints handled by autopilot AUTO mode
            wp = waypoints[0]
            await driver.goto(
                lat=float(wp.get("lat", 0)),
                lon=float(wp.get("lon", 0)),
                alt=float(wp.get("alt", 30.0)),
            )
        else:
            logger.warning(f"MAVLink {entity_id}: unhandled action '{action}'")

    def _telemetry_to_entity(self, telem: Any, vehicle: Dict[str, Any]) -> Dict[str, Any]:
        vehicle_id = vehicle["vehicle_id"]
        battery_pct = float(telem.battery_remaining or 0)

        b = (
            EntityBuilder(f"mavlink-{vehicle_id}", vehicle.get("name", vehicle_id))
            .track()
            .aerial()
            .at(float(telem.lat), float(telem.lon), float(telem.alt))
            .moving(float(telem.heading), float(telem.groundspeed), float(telem.climb_rate))
            .label(vehicle.get("class_label", "uav"))
            .source("mavlink", f"mavlink-{vehicle_id}")
            .org(self.org_id)
            .ttl(10)
            .aerial_telemetry(
                flight_mode=telem.flight_mode,
                battery_pct=battery_pct,
                airspeed_mps=float(telem.airspeed),
                link_quality="good" if telem.gps_fix_type >= 3 else "degraded",
            )
            .meta_dict({
                "vehicle_id": vehicle_id,
                "armed": str(telem.armed).lower(),
                "flight_mode": telem.flight_mode,
                "battery_pct": str(round(battery_pct, 1)),
                "battery_voltage": str(round(float(telem.battery_voltage), 2)),
                "gps_fix": str(telem.gps_fix_type),
                "satellites": str(telem.satellites_visible),
                "protocol": "mavlink",
            })
        )

        if battery_pct < 10.0:
            b = b.critical()
        elif battery_pct < 20.0:
            b = b.warning()
        elif telem.armed:
            b = b.active()
        else:
            b = b.standby()

        return b.build()

    async def _run_simulated_vehicle(self, vehicle: Dict[str, Any]):
        vehicle_id = vehicle["vehicle_id"]

        class _T:
            pass

        telem = _T()
        t = 0.0
        while not self.stopped:
            t += self.publish_interval
            telem.lat = 34.0 + 0.001 * math.sin(t * 0.05)
            telem.lon = -118.0 + 0.001 * math.cos(t * 0.05)
            telem.alt = 120.0 + 20.0 * math.sin(t * 0.02)
            telem.relative_alt = telem.alt
            telem.heading = (t * 2.0) % 360.0
            telem.groundspeed = 12.0 + 3.0 * math.sin(t * 0.1)
            telem.airspeed = 13.0 + 2.0 * math.cos(t * 0.08)
            telem.climb_rate = 0.5 * math.sin(t * 0.15)
            telem.battery_voltage = 22.2 - 0.001 * t
            telem.battery_remaining = max(0, 100 - int(t * 0.05))
            telem.gps_fix_type = 3
            telem.satellites_visible = 14
            telem.flight_mode = "GUIDED"
            telem.armed = True

            entity = self._telemetry_to_entity(telem, vehicle)
            entity["metadata"]["simulated"] = "true"
            self.publish(entity, qos=0)
            await self.sleep(self.publish_interval)
