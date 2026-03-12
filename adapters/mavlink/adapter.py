"""
Summit.OS MAVLink Adapter

Connects to MAVLink-compatible autopilots (ArduPilot, PX4, any MAVLink v2
vehicle) and publishes real-time telemetry as Summit.OS TRACK entities
into the data fabric via MQTT.

MAVLink is the drone protocol — used by DJI (via DroneKit compatibility),
ArduPilot, PX4, and virtually every serious UAV platform. This adapter
bridges the HAL MAVLink driver into the Summit.OS entity stream.

Supports multiple simultaneous vehicles via a vehicle config file.

Vehicle Config (MAVLINK_VEHICLES env var, path to JSON file):
    [
      {
        "vehicle_id": "drone-01",
        "connection_string": "udp:192.168.1.100:14550",
        "system_id": 1,
        "name": "Recon Alpha",
        "class_label": "quadcopter"
      },
      {
        "vehicle_id": "drone-02",
        "connection_string": "serial:/dev/ttyUSB0:57600",
        "system_id": 2,
        "name": "Recon Bravo",
        "class_label": "fixed_wing"
      }
    ]

Environment variables:
    MAVLINK_ENABLED         - "true" to enable (default: "false")
    MAVLINK_VEHICLES        - path to JSON vehicle config file
    MAVLINK_CONNECTION      - single vehicle connection string (if no config file)
                              (default: "udp:127.0.0.1:14550")
    MAVLINK_VEHICLE_ID      - single vehicle ID (default: "drone-01")
    MAVLINK_TELEMETRY_HZ    - publish rate in Hz (default: 2, max: 10)
    MAVLINK_ORG_ID          - org_id for multi-tenant filtering (default: "")
    MQTT_HOST / MQTT_PORT   - broker connection
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("summit.adapter.mavlink")


def _load_vehicle_config(path: Optional[str]) -> List[Dict[str, Any]]:
    """Load vehicle list from JSON file or build single-vehicle config from env."""
    if path:
        try:
            with open(path, "r") as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} vehicle definitions from {path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load vehicle config from {path}: {e}")

    # Single vehicle from env vars
    return [
        {
            "vehicle_id": os.getenv("MAVLINK_VEHICLE_ID", "drone-01"),
            "connection_string": os.getenv("MAVLINK_CONNECTION", "udp:127.0.0.1:14550"),
            "system_id": 1,
            "name": os.getenv("MAVLINK_VEHICLE_ID", "drone-01"),
            "class_label": "uav",
        }
    ]


def _telemetry_to_entity(
    telemetry: Any,
    vehicle: Dict[str, Any],
    org_id: str,
    now_iso: str,
) -> Dict[str, Any]:
    """Convert MAVLink telemetry to a Summit.OS TRACK entity."""
    vehicle_id = vehicle["vehicle_id"]
    name = vehicle.get("name", vehicle_id)
    class_label = vehicle.get("class_label", "uav")

    # Determine state from armed/flight mode
    if telemetry.armed:
        state = "ACTIVE"
    else:
        state = "STANDBY"

    battery_pct = float(telemetry.battery_remaining) if telemetry.battery_remaining else 0.0
    if battery_pct < 10.0:
        state = "CRITICAL"
    elif battery_pct < 20.0:
        state = "WARNING"

    entity_id = f"mavlink-{vehicle_id}"

    return {
        "entity_id": entity_id,
        "id": entity_id,
        "entity_type": "TRACK",
        "domain": "AERIAL",
        "state": state,
        "name": name,
        "class_label": class_label,
        "confidence": 1.0,
        "kinematics": {
            "position": {
                "latitude": float(telemetry.lat),
                "longitude": float(telemetry.lon),
                "altitude_msl": float(telemetry.alt),
                "altitude_agl": float(telemetry.relative_alt),
            },
            "heading_deg": float(telemetry.heading),
            "speed_mps": float(telemetry.groundspeed),
            "climb_rate": float(telemetry.climb_rate),
        },
        "aerial": {
            "altitude_agl": float(telemetry.relative_alt),
            "altitude_msl": float(telemetry.alt),
            "airspeed_mps": float(telemetry.airspeed),
            "flight_mode": telemetry.flight_mode,
            "battery_pct": battery_pct,
            "link_quality": "good" if telemetry.gps_fix_type >= 3 else "degraded",
        },
        "provenance": {
            "source_id": f"mavlink-{vehicle_id}",
            "source_type": "mavlink",
            "org_id": org_id,
            "created_at": time.time(),
            "updated_at": time.time(),
            "version": 1,
        },
        "metadata": {
            "vehicle_id": vehicle_id,
            "armed": str(telemetry.armed).lower(),
            "flight_mode": telemetry.flight_mode,
            "battery_pct": str(round(battery_pct, 1)),
            "battery_voltage": str(round(telemetry.battery_voltage, 2)),
            "gps_fix": str(telemetry.gps_fix_type),
            "satellites": str(telemetry.satellites_visible),
            "roll_deg": str(round(telemetry.roll, 1)),
            "pitch_deg": str(round(telemetry.pitch, 1)),
            "protocol": "mavlink",
        },
        "ttl_seconds": 10,  # short TTL — if telemetry stops, entity expires fast
        "ts": now_iso,
    }


class MAVLinkVehicleWorker:
    """
    Manages one MAVLink vehicle connection and publishes telemetry entities.
    """

    def __init__(
        self,
        vehicle: Dict[str, Any],
        mqtt_client: Any,
        org_id: str,
        publish_interval: float,
    ):
        self.vehicle = vehicle
        self.mqtt = mqtt_client
        self.org_id = org_id
        self.publish_interval = publish_interval
        self._stop = asyncio.Event()
        self._stats = {"published": 0, "errors": 0}

    async def run(self):
        vehicle_id = self.vehicle["vehicle_id"]
        connection_string = self.vehicle.get("connection_string", "udp:127.0.0.1:14550")
        system_id = self.vehicle.get("system_id", 1)

        logger.info(f"MAVLink worker starting: {vehicle_id} @ {connection_string}")

        try:
            import sys
            import os
            # Allow importing HAL driver from packages directory
            packages_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "packages"
            )
            if packages_path not in sys.path:
                sys.path.insert(0, packages_path)

            from hal.drivers.mavlink import MAVLinkDriver

            driver = MAVLinkDriver(
                connection_string=connection_string,
                system_id=system_id,
            )

            connected = await driver.connect()
            if not connected:
                logger.error(f"Could not connect to {vehicle_id} at {connection_string}")
                return

            await driver.start_telemetry()
            logger.info(f"MAVLink {vehicle_id}: telemetry streaming")

            while not self._stop.is_set():
                try:
                    now_iso = datetime.now(timezone.utc).isoformat()
                    entity = _telemetry_to_entity(
                        driver.telemetry, self.vehicle, self.org_id, now_iso
                    )
                    topic = f"entities/{entity['entity_id']}/update"
                    self.mqtt.publish(topic, json.dumps(entity), qos=0)
                    self._stats["published"] += 1
                except Exception as e:
                    self._stats["errors"] += 1
                    logger.debug(f"MAVLink publish error for {vehicle_id}: {e}")

                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.publish_interval)
                    break
                except asyncio.TimeoutError:
                    pass

            await driver.disconnect()

        except ImportError:
            logger.warning(f"HAL MAVLink driver unavailable for {vehicle_id} — running simulated")
            await self._run_simulated()
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"MAVLink worker error for {vehicle_id}: {e}")

        logger.info(f"MAVLink worker stopped: {vehicle_id} (stats={self._stats})")

    async def _run_simulated(self):
        """Publish simulated MAVLink telemetry when pymavlink is unavailable."""
        import math

        vehicle_id = self.vehicle["vehicle_id"]

        class _SimTelemetry:
            pass

        telem = _SimTelemetry()
        t = 0

        while not self._stop.is_set():
            t += self.publish_interval
            telem.lat = 34.0 + 0.001 * math.sin(t * 0.05)
            telem.lon = -118.0 + 0.001 * math.cos(t * 0.05)
            telem.alt = 120.0 + 20.0 * math.sin(t * 0.02)
            telem.relative_alt = 120.0 + 20.0 * math.sin(t * 0.02)
            telem.heading = (t * 2.0) % 360.0
            telem.groundspeed = 12.0 + 3.0 * math.sin(t * 0.1)
            telem.airspeed = 13.0 + 2.0 * math.cos(t * 0.08)
            telem.climb_rate = 0.5 * math.sin(t * 0.15)
            telem.roll = 5.0 * math.sin(t * 0.3)
            telem.pitch = 3.0 * math.cos(t * 0.25)
            telem.yaw = (t * 2.0) % 360.0
            telem.battery_voltage = 22.2 - 0.001 * t
            telem.battery_remaining = max(0, 100 - int(t * 0.05))
            telem.gps_fix_type = 3
            telem.satellites_visible = 14
            telem.flight_mode = "GUIDED"
            telem.armed = True

            now_iso = datetime.now(timezone.utc).isoformat()
            entity = _telemetry_to_entity(telem, self.vehicle, self.org_id, now_iso)
            entity["metadata"]["simulated"] = "true"
            topic = f"entities/{entity['entity_id']}/update"
            self.mqtt.publish(topic, json.dumps(entity), qos=0)
            self._stats["published"] += 1

            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.publish_interval)
                break
            except asyncio.TimeoutError:
                pass

    async def stop(self):
        self._stop.set()


class MAVLinkAdapter:
    """
    Manages multiple MAVLink vehicle workers, each publishing telemetry
    as Summit.OS TRACK entities into the data fabric.
    """

    def __init__(
        self,
        mqtt_client: Any,
        vehicles_path: Optional[str] = os.getenv("MAVLINK_VEHICLES"),
        org_id: str = os.getenv("MAVLINK_ORG_ID", ""),
        telemetry_hz: float = min(float(os.getenv("MAVLINK_TELEMETRY_HZ", "2")), 10.0),
    ):
        self.mqtt = mqtt_client
        self.org_id = org_id
        self.publish_interval = 1.0 / max(telemetry_hz, 0.1)
        self.vehicles = _load_vehicle_config(vehicles_path)
        self._workers: List[MAVLinkVehicleWorker] = []
        self._stop = asyncio.Event()

    @property
    def enabled(self) -> bool:
        return os.getenv("MAVLINK_ENABLED", "false").lower() == "true"

    async def start(self):
        if not self.enabled:
            logger.info("MAVLink adapter disabled")
            return

        logger.info(
            f"MAVLink adapter starting ({len(self.vehicles)} vehicle(s), "
            f"{1.0/self.publish_interval:.1f}Hz)"
        )

        self._workers = [
            MAVLinkVehicleWorker(v, self.mqtt, self.org_id, self.publish_interval)
            for v in self.vehicles
        ]

        tasks = [asyncio.create_task(w.run()) for w in self._workers]
        await self._stop.wait()

        for w in self._workers:
            await w.stop()
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("MAVLink adapter stopped")

    async def stop(self):
        self._stop.set()
