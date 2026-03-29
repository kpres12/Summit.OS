"""
Summit.OS — Tesla Fleet API Adapter
=====================================

Integrates Tesla vehicles into Summit.OS via the official Tesla Fleet API.
Tesla publishes and maintains this API — it is fully open and documented.

Use cases
---------
- Autonomous logistics: track vehicle location, dispatch to waypoint
- Fleet coordination: monitor range, charging state, occupancy
- Emergency response: know where your vehicles are at all times

Capabilities
------------
- Vehicle location (lat/lon, heading, speed)
- State of charge / range
- Drive state (parked, driving, charging)
- Climate state
- Optional: wake vehicle, set navigation destination (requires WRITE capability)

Dependencies
------------
    pip install teslapy

Config extras
-------------
email           : str   — Tesla account email
vehicle_id      : str   — Tesla vehicle VIN or internal ID (leave blank for all vehicles)
refresh_token   : str   — Tesla OAuth refresh token (from initial auth flow)
poll_interval_seconds : float — how often to poll (default 30.0; Tesla rate-limits)
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("summit.adapters.tesla")

try:
    import teslapy
    _TESLAPY_AVAILABLE = True
except ImportError:
    teslapy = None  # type: ignore
    _TESLAPY_AVAILABLE = False


_DRIVE_STATE_MAP = {
    "Parked": "parked",
    "Driving": "driving",
    "Charging": "charging",
    "Asleep": "asleep",
    "Offline": "offline",
    "Unknown": "unknown",
}


class TeslaAdapter(BaseAdapter):
    """
    Polls Tesla Fleet API and emits GROUND_VEHICLE observations.

    Each vehicle in the account becomes a separate entity in Summit.OS.
    Rate-limit aware: Tesla allows ~200 API calls/day per vehicle when sleeping.
    Set poll_interval_seconds >= 30 to stay well under limits.
    """

    adapter_type = "tesla"

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        ex = config.extra
        self._email: str = ex.get("email", "")
        self._vin_filter: Optional[str] = ex.get("vehicle_id") or ex.get("vin")
        self._tesla: Optional[object] = None
        self._vehicles: list = []

    async def connect(self) -> None:
        if not _TESLAPY_AVAILABLE:
            raise RuntimeError(
                "teslapy not installed. Run: pip install teslapy"
            )
        if not self._email:
            raise ValueError("Tesla adapter requires 'email' in config.extra")

        loop = asyncio.get_event_loop()

        def _auth():
            t = teslapy.Tesla(self._email)
            if not t.authorized:
                raise RuntimeError(
                    "Tesla not authorized. Run teslapy auth flow first: "
                    "https://github.com/tdorssers/TeslaPy#authorization"
                )
            vehicles = t.vehicle_list()
            if self._vin_filter:
                vehicles = [v for v in vehicles if v.get("vin") == self._vin_filter]
            return t, vehicles

        self._tesla, self._vehicles = await loop.run_in_executor(None, _auth)
        logger.info("Tesla connected: %d vehicle(s)", len(self._vehicles))

    async def disconnect(self) -> None:
        if self._tesla:
            try:
                self._tesla.close()
            except Exception:
                pass
        self._tesla = None
        self._vehicles = []

    async def stream_observations(self) -> AsyncIterator[dict]:
        loop = asyncio.get_event_loop()
        while not self._stop_event.is_set():
            for vehicle in self._vehicles:
                try:
                    obs = await loop.run_in_executor(None, self._poll_vehicle, vehicle)
                    if obs:
                        yield obs
                except Exception as e:
                    logger.debug("Tesla poll failed for %s: %s", vehicle.get("vin", "?"), e)
            await asyncio.sleep(self.config.poll_interval_seconds)

    def _poll_vehicle(self, vehicle) -> Optional[dict]:
        try:
            vehicle.get_vehicle_summary()
            state = vehicle.get("state", "")
            if state in ("asleep", "offline"):
                # Don't wake sleeping vehicles — Tesla charges API calls
                return self._build_obs(vehicle, None, None)
            data = vehicle.get_vehicle_data()
            return self._build_obs(vehicle, data.get("drive_state"), data.get("charge_state"))
        except Exception as e:
            logger.debug("Vehicle data error: %s", e)
            return None

    def _build_obs(self, vehicle, drive_state, charge_state) -> dict:
        now = datetime.now(timezone.utc)
        vin = vehicle.get("vin", self.config.adapter_id)
        display_name = vehicle.get("display_name") or vehicle.get("vin", "Tesla")

        obs: dict = {
            "source_id": f"{vin}:{now.timestamp():.3f}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": vin,
            "callsign": display_name,
            "entity_type": "GROUND_VEHICLE",
            "classification": "tesla_ev",
            "ts_iso": now.isoformat(),
            "metadata": {
                "vin": vin,
                "vehicle_state": vehicle.get("state", "unknown"),
                "model": vehicle.get("model", ""),
            },
        }

        if drive_state:
            lat = drive_state.get("latitude")
            lon = drive_state.get("longitude")
            if lat and lon:
                obs["position"] = {"lat": lat, "lon": lon, "alt_m": None}
            heading = drive_state.get("heading")
            speed = drive_state.get("speed")  # mph
            if speed is not None:
                speed_mps = speed * 0.44704
            else:
                speed_mps = None
            obs["velocity"] = {
                "heading_deg": heading,
                "speed_mps": round(speed_mps, 2) if speed_mps is not None else None,
                "vertical_mps": None,
            }
            obs["metadata"]["shift_state"] = drive_state.get("shift_state", "P")

        if charge_state:
            obs["metadata"]["battery_pct"] = charge_state.get("battery_level")
            obs["metadata"]["est_range_miles"] = charge_state.get("est_battery_range")
            obs["metadata"]["charging_state"] = charge_state.get("charging_state", "")

        return obs
