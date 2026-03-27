"""
Summit.OS — Modbus TCP Adapter
================================

Connects to industrial sensors, PLCs, and SCADA systems via Modbus TCP.
Reads configured holding registers and coils at a configurable poll interval
and emits INDUSTRIAL_SENSOR entities.

Dependencies
------------
    pip install pymodbus>=3.6.0
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

try:
    from pymodbus.client import AsyncModbusTcpClient
    from pymodbus.exceptions import ModbusException
except ImportError:
    raise ImportError(
        "pymodbus is required for ModbusAdapter. Install with: pip install pymodbus>=3.6.0"
    )

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("summit.adapters.modbus")


class ModbusAdapter(BaseAdapter):
    """
    Reads Modbus TCP registers and emits INDUSTRIAL_SENSOR observations.

    Config extras
    -------------
    host                  : str
    port                  : int    (default 502)
    unit_id               : int    (default 1)
    poll_interval_seconds : float  (default 5.0)
    entity_lat            : float  (default 0.0)
    entity_lon            : float  (default 0.0)
    registers             : list of register definition dicts, each with:
        address : int
        type    : "holding" | "coil" | "input" | "discrete"
        name    : str
        scale   : float  (optional, default 1.0)
        offset  : float  (optional, default 0.0)
        unit    : str    (optional, e.g. "°C", "bar")
    """

    adapter_type = "modbus"

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        ex = config.extra

        self._host: str = ex.get("host", "")
        if not self._host:
            raise ValueError("host must be set in adapter extra config")
        self._port: int = int(ex.get("port", 502))
        self._unit_id: int = int(ex.get("unit_id", 1))
        self._poll_interval: float = float(
            ex.get("poll_interval_seconds", config.poll_interval_seconds or 5.0)
        )
        self._entity_lat: float = float(ex.get("entity_lat", 0.0))
        self._entity_lon: float = float(ex.get("entity_lon", 0.0))
        self._registers: list[dict] = ex.get("registers", [])

        if not self._registers:
            raise ValueError("At least one register must be configured")

        # Validate register definitions
        for reg in self._registers:
            if "address" not in reg or "name" not in reg:
                raise ValueError(f"Each register must have 'address' and 'name': {reg}")
            reg_type = reg.get("type", "holding")
            if reg_type not in ("holding", "coil", "input", "discrete"):
                raise ValueError(
                    f"Register type must be holding/coil/input/discrete, got: {reg_type!r}"
                )

        self._client: Optional[AsyncModbusTcpClient] = None

    async def connect(self) -> None:
        self._client = AsyncModbusTcpClient(self._host, port=self._port)
        connected = await self._client.connect()
        if not connected:
            raise ConnectionError(
                f"Could not connect to Modbus TCP {self._host}:{self._port}"
            )
        self._log.info(
            "Modbus TCP connected to %s:%d (unit=%d)",
            self._host,
            self._port,
            self._unit_id,
        )

    async def disconnect(self) -> None:
        try:
            if self._client is not None:
                self._client.close()
                self._client = None
        except Exception:
            pass

    async def stream_observations(self) -> AsyncIterator[dict]:
        while not self._stop_event.is_set():
            try:
                register_values = await self._read_all_registers()
                yield self._build_observation(register_values)
            except ModbusException as exc:
                raise RuntimeError(f"Modbus read error: {exc}") from exc
            await self._interruptible_sleep(self._poll_interval)

    async def _read_all_registers(self) -> dict:
        """Read all configured registers and return a name→value dict."""
        values: dict = {}

        for reg in self._registers:
            address: int = int(reg["address"])
            reg_type: str = reg.get("type", "holding")
            name: str = reg["name"]
            scale: float = float(reg.get("scale", 1.0))
            offset: float = float(reg.get("offset", 0.0))
            unit: str = reg.get("unit", "")

            try:
                raw_value = await self._read_register(address, reg_type)
                if raw_value is None:
                    values[name] = None
                else:
                    if reg_type in ("coil", "discrete"):
                        # Coils and discrete inputs are boolean
                        values[name] = bool(raw_value)
                    else:
                        scaled = float(raw_value) * scale + offset
                        values[name] = round(scaled, 6)
                        if unit:
                            values[f"{name}_unit"] = unit
            except Exception as exc:
                self._log.warning(
                    "Failed to read register %s (addr=%d, type=%s): %s",
                    name,
                    address,
                    reg_type,
                    exc,
                )
                values[name] = None

        return values

    async def _read_register(self, address: int, reg_type: str) -> Optional[int]:
        """Read a single register and return its raw integer value."""
        if reg_type == "holding":
            result = await self._client.read_holding_registers(
                address, count=1, slave=self._unit_id
            )
            if result.isError():
                raise ModbusException(f"Holding register read error at {address}")
            return result.registers[0]

        elif reg_type == "input":
            result = await self._client.read_input_registers(
                address, count=1, slave=self._unit_id
            )
            if result.isError():
                raise ModbusException(f"Input register read error at {address}")
            return result.registers[0]

        elif reg_type == "coil":
            result = await self._client.read_coils(
                address, count=1, slave=self._unit_id
            )
            if result.isError():
                raise ModbusException(f"Coil read error at {address}")
            return result.bits[0]

        elif reg_type == "discrete":
            result = await self._client.read_discrete_inputs(
                address, count=1, slave=self._unit_id
            )
            if result.isError():
                raise ModbusException(f"Discrete input read error at {address}")
            return result.bits[0]

        else:
            raise ValueError(f"Unknown register type: {reg_type!r}")

    def _build_observation(self, register_values: dict) -> dict:
        now = datetime.now(timezone.utc)
        entity_id = self.config.adapter_id
        return {
            "source_id": f"{entity_id}:{now.timestamp():.3f}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": entity_id,
            "callsign": self.config.display_name or entity_id,
            "position": {
                "lat": self._entity_lat,
                "lon": self._entity_lon,
                "alt_m": None,
            },
            "velocity": None,
            "entity_type": "INDUSTRIAL_SENSOR",
            "classification": None,
            "metadata": {
                **register_values,
                "host": self._host,
                "port": self._port,
                "unit_id": self._unit_id,
            },
            "ts_iso": now.isoformat(),
        }
