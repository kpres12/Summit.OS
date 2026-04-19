"""
Heli.OS Modbus/TCP Adapter

Polls industrial hardware via Modbus/TCP and publishes each register
as a Heli.OS ASSET Entity into the data fabric.

Modbus is the lingua franca of industrial automation — PLCs, pumps,
pressure sensors, flow meters, HVAC, and power meters.

Register Map Config (MODBUS_REGISTER_MAP env var, path to JSON file):
    [
      {
        "address": 40001,
        "register_type": "holding",
        "name": "PressureValve_01",
        "class_label": "pressure_sensor",
        "unit": "PSI",
        "scale": 0.1,
        "offset": 0.0,
        "domain": "GROUND",
        "warn_above": 800.0,
        "critical_above": 950.0,
        "warn_below": null,
        "critical_below": null
      }
    ]

Environment variables:
    MODBUS_ENABLED          - "true" to enable (default: "false")
    MODBUS_HOST             - PLC/device host (default: "localhost")
    MODBUS_PORT             - Modbus TCP port (default: 502)
    MODBUS_DEVICE_ID        - logical device name (default: "modbus-device-01")
    MODBUS_UNIT_ID          - Modbus unit/slave ID (default: 1)
    MODBUS_POLL_INTERVAL    - seconds between polls (default: 5)
    MODBUS_REGISTER_MAP     - path to JSON register map file
    MODBUS_ORG_ID           - org_id for multi-tenant filtering (default: "")
    MQTT_HOST / MQTT_PORT   - broker connection
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))

from sdk import BaseAdapter, AdapterManifest, EntityBuilder, Protocol, Capability

logger = logging.getLogger("summit.adapter.modbus")

DEFAULT_REGISTER_MAP: List[Dict[str, Any]] = [
    {"address": 40001, "register_type": "holding", "name": "Inlet_Pressure",
     "class_label": "pressure_sensor", "unit": "PSI", "scale": 0.1, "offset": 0.0,
     "domain": "GROUND", "warn_above": 800.0, "critical_above": 950.0, "critical_below": 10.0},
    {"address": 40002, "register_type": "holding", "name": "Outlet_Pressure",
     "class_label": "pressure_sensor", "unit": "PSI", "scale": 0.1, "offset": 0.0,
     "domain": "GROUND", "warn_above": 750.0, "critical_above": 900.0, "critical_below": 5.0},
    {"address": 40003, "register_type": "holding", "name": "Flow_Rate",
     "class_label": "flow_meter", "unit": "L/min", "scale": 0.01, "offset": 0.0,
     "domain": "GROUND", "warn_above": 5000.0, "warn_below": 10.0, "critical_below": 0.0},
    {"address": 40004, "register_type": "holding", "name": "Motor_Temperature",
     "class_label": "temperature_sensor", "unit": "degC", "scale": 0.1, "offset": 0.0,
     "domain": "GROUND", "warn_above": 80.0, "critical_above": 100.0},
    {"address": 1, "register_type": "coil", "name": "Main_Valve",
     "class_label": "valve", "unit": "bool", "scale": 1.0, "offset": 0.0, "domain": "GROUND"},
]


def _load_register_map(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return DEFAULT_REGISTER_MAP
    try:
        with open(path) as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} register definitions from {path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load register map from {path}: {e} — using demo map")
        return DEFAULT_REGISTER_MAP


class ModbusAdapter(BaseAdapter):
    """Polls a Modbus/TCP device and publishes register values as ASSET entities."""

    MANIFEST = AdapterManifest(
        name="modbus",
        version="1.0.0",
        protocol=Protocol.MODBUS,
        capabilities=[Capability.READ],
        entity_types=["ASSET"],
        description="Industrial Modbus/TCP adapter — PLCs, pumps, sensors, valves",
        required_env=["MODBUS_HOST"],
        optional_env=["MODBUS_PORT", "MODBUS_UNIT_ID", "MODBUS_REGISTER_MAP", "MODBUS_POLL_INTERVAL"],
    )

    def __init__(
        self,
        host: str = os.getenv("MODBUS_HOST", "localhost"),
        port: int = int(os.getenv("MODBUS_PORT", "502")),
        device_id: str = os.getenv("MODBUS_DEVICE_ID", "modbus-device-01"),
        unit_id: int = int(os.getenv("MODBUS_UNIT_ID", "1")),
        poll_interval: float = float(os.getenv("MODBUS_POLL_INTERVAL", "5")),
        register_map_path: Optional[str] = os.getenv("MODBUS_REGISTER_MAP"),
        org_id: str = os.getenv("MODBUS_ORG_ID", ""),
        **kwargs,
    ):
        super().__init__(device_id=device_id, org_id=org_id, **kwargs)
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.poll_interval = max(poll_interval, 1.0)
        self.registers = _load_register_map(register_map_path)
        self._stats = {"polls": 0, "published": 0, "errors": 0}

    @property
    def enabled(self) -> bool:
        return os.getenv("MODBUS_ENABLED", "false").lower() == "true"

    async def run(self):
        logger.info(
            f"Modbus adapter running (host={self.host}:{self.port}, "
            f"device={self.device_id}, registers={len(self.registers)}, "
            f"interval={self.poll_interval}s)"
        )
        while not self.stopped:
            try:
                await self._poll()
            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"Modbus poll error: {e}")
            await self.sleep(self.poll_interval)

    async def _poll(self):
        try:
            from pymodbus.client import AsyncModbusTcpClient
        except ImportError:
            logger.warning("pymodbus not installed — publishing simulated Modbus data")
            await self._poll_simulated()
            return

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
                    entity = self._reg_to_entity(reg, raw)
                    self.publish(entity, qos=1)
                    published += 1
                except Exception as e:
                    logger.debug(f"Register {reg.get('address')} read error: {e}")

        self._stats["polls"] += 1
        self._stats["published"] += published
        logger.info(f"Modbus: published {published}/{len(self.registers)} registers from {self.device_id}")

    async def _read_register(self, client: Any, reg: Dict[str, Any]) -> Optional[Any]:
        address = reg["address"]
        reg_type = reg.get("register_type", "holding")
        count = reg.get("count", 1)
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
            return None

        if result.isError():
            return None
        return result.bits[0] if reg_type in ("coil", "discrete") else result.registers[0]

    def _reg_to_entity(self, reg: Dict[str, Any], raw: Any) -> Dict[str, Any]:
        name = reg["name"]
        reg_type = reg.get("register_type", "holding")
        unit = reg.get("unit", "")
        scale = float(reg.get("scale", 1.0))
        offset = float(reg.get("offset", 0.0))

        scaled = (1.0 if raw else 0.0) if reg_type in ("coil", "discrete") else float(raw) * scale + offset

        b = (
            EntityBuilder(
                f"modbus-{self.device_id}-{name.lower().replace(' ', '-')}",
                f"{self.device_id}/{name}",
            )
            .asset()
            .ground()
            .label(reg.get("class_label", "sensor"))
            .value(scaled, unit)
            .source("modbus", f"modbus-{self.device_id}")
            .org(self.org_id)
            .ttl(30)
            .meta_dict({
                "raw_value": str(raw),
                "register_address": str(reg["address"]),
                "register_type": reg_type,
                "device_id": self.device_id,
                "protocol": "modbus",
            })
        )

        if reg.get("warn_above") is not None:
            b = b.warn_above(float(reg["warn_above"]))
        if reg.get("critical_above") is not None:
            b = b.critical_above(float(reg["critical_above"]))
        if reg.get("warn_below") is not None:
            b = b.warn_below(float(reg["warn_below"]))
        if reg.get("critical_below") is not None:
            b = b.critical_below(float(reg["critical_below"]))

        return b.build()

    async def _poll_simulated(self):
        t = time.time()
        for reg in self.registers:
            reg_type = reg.get("register_type", "holding")
            raw = (int(500 + 100 * math.sin(t * 0.1)) if reg_type in ("holding", "input")
                   else bool(int(t) % 10 < 7))
            entity = self._reg_to_entity(reg, raw)
            entity["metadata"]["simulated"] = "true"
            self.publish(entity, qos=1)
        self._stats["polls"] += 1
        logger.debug(f"Modbus (simulated): published {len(self.registers)} entities")
