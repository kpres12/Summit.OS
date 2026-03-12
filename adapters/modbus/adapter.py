"""
Summit.OS Modbus/TCP Adapter

Polls industrial hardware via Modbus/TCP and publishes each register
as a Summit.OS Entity into the data fabric via MQTT.

Modbus is the lingua franca of industrial automation — PLCs, pumps,
pressure sensors, flow meters, HVAC, and power meters all speak it.
If Summit.OS can read Modbus, it can talk to the majority of the
world's industrial infrastructure.

Register Map Config (MODBUS_REGISTER_MAP env var, path to JSON file):
    [
      {
        "address": 40001,
        "register_type": "holding",   // "holding", "input", "coil", "discrete"
        "name": "PressureValve_01",
        "class_label": "pressure_sensor",
        "unit": "PSI",
        "scale": 0.1,                 // multiply raw value by this
        "offset": 0.0,                // add after scaling
        "domain": "GROUND",
        "warn_above": 800.0,          // value that sets state -> WARNING
        "critical_above": 950.0,      // value that sets state -> CRITICAL
        "warn_below": null,
        "critical_below": null
      }
    ]

Environment variables:
    MODBUS_ENABLED          - "true" to enable (default: "false")
    MODBUS_HOST             - PLC/device host (default: "localhost")
    MODBUS_PORT             - Modbus TCP port (default: 502)
    MODBUS_DEVICE_ID        - logical device name used in entity IDs (default: "modbus-device-01")
    MODBUS_UNIT_ID          - Modbus unit/slave ID (default: 1)
    MODBUS_POLL_INTERVAL    - seconds between polls (default: 5)
    MODBUS_REGISTER_MAP     - path to JSON register map file (default: built-in demo map)
    MODBUS_ORG_ID           - org_id for multi-tenant filtering (default: "")
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

logger = logging.getLogger("summit.adapter.modbus")

# Default demo register map — describes a hypothetical pumping station.
# Operators replace this with their actual register map JSON file.
DEFAULT_REGISTER_MAP: List[Dict[str, Any]] = [
    {
        "address": 40001,
        "register_type": "holding",
        "name": "Inlet_Pressure",
        "class_label": "pressure_sensor",
        "unit": "PSI",
        "scale": 0.1,
        "offset": 0.0,
        "domain": "GROUND",
        "warn_above": 800.0,
        "critical_above": 950.0,
        "warn_below": None,
        "critical_below": 10.0,
    },
    {
        "address": 40002,
        "register_type": "holding",
        "name": "Outlet_Pressure",
        "class_label": "pressure_sensor",
        "unit": "PSI",
        "scale": 0.1,
        "offset": 0.0,
        "domain": "GROUND",
        "warn_above": 750.0,
        "critical_above": 900.0,
        "warn_below": None,
        "critical_below": 5.0,
    },
    {
        "address": 40003,
        "register_type": "holding",
        "name": "Flow_Rate",
        "class_label": "flow_meter",
        "unit": "L/min",
        "scale": 0.01,
        "offset": 0.0,
        "domain": "GROUND",
        "warn_above": 5000.0,
        "critical_above": None,
        "warn_below": 10.0,
        "critical_below": 0.0,
    },
    {
        "address": 40004,
        "register_type": "holding",
        "name": "Motor_Temperature",
        "class_label": "temperature_sensor",
        "unit": "degC",
        "scale": 0.1,
        "offset": 0.0,
        "domain": "GROUND",
        "warn_above": 80.0,
        "critical_above": 100.0,
        "warn_below": None,
        "critical_below": None,
    },
    {
        "address": 1,
        "register_type": "coil",
        "name": "Main_Valve",
        "class_label": "valve",
        "unit": "bool",
        "scale": 1.0,
        "offset": 0.0,
        "domain": "GROUND",
        "warn_above": None,
        "critical_above": None,
        "warn_below": None,
        "critical_below": None,
    },
]


def _load_register_map(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        logger.info("No register map file specified — using built-in demo map")
        return DEFAULT_REGISTER_MAP
    try:
        with open(path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} register definitions from {path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load register map from {path}: {e} — using demo map")
        return DEFAULT_REGISTER_MAP


def _compute_state(value: float, reg: Dict[str, Any]) -> str:
    """Map a scaled register value to a Summit entity state."""
    critical_above = reg.get("critical_above")
    critical_below = reg.get("critical_below")
    warn_above = reg.get("warn_above")
    warn_below = reg.get("warn_below")

    if critical_above is not None and value >= critical_above:
        return "CRITICAL"
    if critical_below is not None and value <= critical_below:
        return "CRITICAL"
    if warn_above is not None and value >= warn_above:
        return "WARNING"
    if warn_below is not None and value <= warn_below:
        return "WARNING"
    return "ACTIVE"


def _register_to_entity(
    reg: Dict[str, Any],
    raw_value: Any,
    device_id: str,
    org_id: str,
    now_iso: str,
) -> Dict[str, Any]:
    """Convert a raw Modbus register value to a Summit.OS Entity dict."""
    name = reg["name"]
    reg_type = reg.get("register_type", "holding")
    unit = reg.get("unit", "")
    scale = float(reg.get("scale", 1.0))
    offset = float(reg.get("offset", 0.0))
    domain = reg.get("domain", "GROUND")
    address = reg["address"]

    # Scale the raw value
    if reg_type == "coil" or reg_type == "discrete":
        scaled = 1.0 if raw_value else 0.0
    else:
        scaled = float(raw_value) * scale + offset

    state = _compute_state(scaled, reg)

    entity_id = f"modbus-{device_id}-{name.lower().replace(' ', '-')}"

    return {
        "entity_id": entity_id,
        "id": entity_id,
        "entity_type": "ASSET",
        "domain": domain,
        "state": state,
        "name": f"{device_id}/{name}",
        "class_label": reg.get("class_label", "sensor"),
        "confidence": 1.0,
        "kinematics": {
            "position": {
                "latitude": 0.0,
                "longitude": 0.0,
                "altitude_msl": 0.0,
                "altitude_agl": 0.0,
            },
            "heading_deg": 0.0,
            "speed_mps": 0.0,
            "climb_rate": 0.0,
        },
        "provenance": {
            "source_id": f"modbus-{device_id}",
            "source_type": "modbus",
            "org_id": org_id,
            "created_at": time.time(),
            "updated_at": time.time(),
            "version": 1,
        },
        "metadata": {
            "value": str(round(scaled, 4)),
            "raw_value": str(raw_value),
            "unit": unit,
            "register_address": str(address),
            "register_type": reg_type,
            "device_id": device_id,
            "protocol": "modbus",
            "state_reason": state,
        },
        "ttl_seconds": 30,
        "ts": now_iso,
    }


class ModbusAdapter:
    """
    Polls a Modbus/TCP device and publishes register values as
    Summit.OS ASSET entities to MQTT.
    """

    def __init__(
        self,
        mqtt_client: Any,
        host: str = os.getenv("MODBUS_HOST", "localhost"),
        port: int = int(os.getenv("MODBUS_PORT", "502")),
        device_id: str = os.getenv("MODBUS_DEVICE_ID", "modbus-device-01"),
        unit_id: int = int(os.getenv("MODBUS_UNIT_ID", "1")),
        poll_interval: float = float(os.getenv("MODBUS_POLL_INTERVAL", "5")),
        register_map_path: Optional[str] = os.getenv("MODBUS_REGISTER_MAP"),
        org_id: str = os.getenv("MODBUS_ORG_ID", ""),
    ):
        self.mqtt = mqtt_client
        self.host = host
        self.port = port
        self.device_id = device_id
        self.unit_id = unit_id
        self.poll_interval = max(poll_interval, 1.0)
        self.org_id = org_id
        self.registers = _load_register_map(register_map_path)
        self._stop = asyncio.Event()
        self._stats = {"polls": 0, "published": 0, "errors": 0}

    @property
    def enabled(self) -> bool:
        return os.getenv("MODBUS_ENABLED", "false").lower() == "true"

    async def start(self):
        if not self.enabled:
            logger.info("Modbus adapter disabled")
            return

        logger.info(
            f"Modbus adapter starting (host={self.host}:{self.port}, "
            f"device={self.device_id}, unit={self.unit_id}, "
            f"registers={len(self.registers)}, interval={self.poll_interval}s)"
        )

        while not self._stop.is_set():
            try:
                await self._poll()
            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"Modbus poll error: {e}")

            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.poll_interval)
                break
            except asyncio.TimeoutError:
                pass

        logger.info(f"Modbus adapter stopped (stats={self._stats})")

    async def stop(self):
        self._stop.set()

    async def _poll(self):
        """Connect to device, read all registers, publish entities."""
        try:
            from pymodbus.client import AsyncModbusTcpClient
            from pymodbus.exceptions import ModbusException
        except ImportError:
            logger.warning("pymodbus not installed — publishing simulated Modbus data")
            await self._poll_simulated()
            return

        now_iso = datetime.now(timezone.utc).isoformat()
        published = 0

        async with AsyncModbusTcpClient(self.host, port=self.port) as client:
            if not client.connected:
                logger.error(f"Could not connect to Modbus device at {self.host}:{self.port}")
                self._stats["errors"] += 1
                return

            for reg in self.registers:
                try:
                    raw = await self._read_register(client, reg)
                    if raw is None:
                        continue
                    entity = _register_to_entity(
                        reg, raw, self.device_id, self.org_id, now_iso
                    )
                    topic = f"entities/{entity['entity_id']}/update"
                    self.mqtt.publish(topic, json.dumps(entity), qos=1)
                    published += 1
                except Exception as e:
                    logger.debug(f"Register {reg.get('address')} read error: {e}")

        self._stats["polls"] += 1
        self._stats["published"] += published
        logger.info(f"Modbus: published {published}/{len(self.registers)} registers from {self.device_id}")

    async def _read_register(self, client: Any, reg: Dict[str, Any]) -> Optional[Any]:
        """Read a single register and return its raw value."""
        address = reg["address"]
        reg_type = reg.get("register_type", "holding")
        count = reg.get("count", 1)

        # Modbus addressing: holding registers are 1-indexed in convention
        # but pymodbus uses 0-based. Normalize 4xxxx -> 0-based.
        addr = (address % 10000) - 1 if address >= 10000 else address

        if reg_type == "holding":
            result = await client.read_holding_registers(addr, count=count, slave=self.unit_id)
        elif reg_type == "input":
            result = await client.read_input_registers(addr, count=count, slave=self.unit_id)
        elif reg_type == "coil":
            result = await client.read_coils(addr, count=count, slave=self.unit_id)
        elif reg_type == "discrete":
            result = await client.read_discrete_inputs(addr, count=count, slave=self.unit_id)
        else:
            logger.warning(f"Unknown register type: {reg_type}")
            return None

        if result.isError():
            logger.debug(f"Modbus error reading {reg_type}@{address}: {result}")
            return None

        if reg_type in ("coil", "discrete"):
            return result.bits[0]
        else:
            return result.registers[0]

    async def _poll_simulated(self):
        """Publish simulated register values when pymodbus is unavailable."""
        import math
        now_iso = datetime.now(timezone.utc).isoformat()
        t = time.time()
        published = 0

        sim_values = {
            "holding": int(500 + 100 * math.sin(t * 0.1)),   # oscillating analog
            "input": int(300 + 50 * math.cos(t * 0.05)),
            "coil": bool(int(t) % 10 < 7),                    # mostly open
            "discrete": True,
        }

        for reg in self.registers:
            reg_type = reg.get("register_type", "holding")
            raw = sim_values.get(reg_type, 0)
            entity = _register_to_entity(
                reg, raw, self.device_id, self.org_id, now_iso
            )
            entity["metadata"]["simulated"] = "true"
            topic = f"entities/{entity['entity_id']}/update"
            self.mqtt.publish(topic, json.dumps(entity), qos=1)
            published += 1

        self._stats["polls"] += 1
        self._stats["published"] += published
        logger.info(f"Modbus (simulated): published {published} entities")
