"""
EntityBuilder — fluent builder for Summit.OS entities.

Eliminates the boilerplate of constructing entity dicts by hand.
Every adapter should use this to guarantee schema consistency.

Usage:
    entity = (
        EntityBuilder("modbus-pump-01", "Inlet Pressure")
        .asset()
        .ground()
        .at(lat=34.05, lon=-118.25)
        .value(255.0, "PSI")
        .warn_above(800.0)
        .critical_above(950.0)
        .source("modbus", "pump-station-01")
        .build()
    )
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class EntityBuilder:
    """Fluent builder for Summit.OS entity dicts."""

    def __init__(self, entity_id: str, name: str):
        self._id = entity_id
        self._name = name
        self._entity_type = "ASSET"
        self._domain = "GROUND"
        self._state = "ACTIVE"
        self._class_label = "sensor"
        self._confidence = 1.0
        self._lat = 0.0
        self._lon = 0.0
        self._alt = 0.0
        self._heading = 0.0
        self._speed = 0.0
        self._climb = 0.0
        self._source_id = ""
        self._source_type = "custom"
        self._org_id = ""
        self._ttl = 60
        self._metadata: Dict[str, str] = {}
        self._aerial: Optional[Dict[str, Any]] = None
        self._value: Optional[float] = None
        self._unit: str = ""

    # ── Entity type ──────────────────────────────────────────────────────────

    def asset(self) -> "EntityBuilder":
        """Stationary or slow-moving physical asset (sensor, valve, etc.)."""
        self._entity_type = "ASSET"
        return self

    def track(self) -> "EntityBuilder":
        """Moving tracked object (drone, vehicle, aircraft)."""
        self._entity_type = "TRACK"
        return self

    def alert(self) -> "EntityBuilder":
        self._entity_type = "ALERT"
        return self

    # ── Domain ───────────────────────────────────────────────────────────────

    def ground(self) -> "EntityBuilder":
        self._domain = "GROUND"
        return self

    def aerial(self) -> "EntityBuilder":
        self._domain = "AERIAL"
        return self

    def maritime(self) -> "EntityBuilder":
        self._domain = "MARITIME"
        return self

    def cyber(self) -> "EntityBuilder":
        self._domain = "CYBER"
        return self

    # ── State ────────────────────────────────────────────────────────────────

    def active(self) -> "EntityBuilder":
        self._state = "ACTIVE"
        return self

    def standby(self) -> "EntityBuilder":
        self._state = "STANDBY"
        return self

    def warning(self) -> "EntityBuilder":
        self._state = "WARNING"
        return self

    def critical(self) -> "EntityBuilder":
        self._state = "CRITICAL"
        return self

    # ── Auto-state from thresholds ───────────────────────────────────────────

    def warn_above(self, threshold: float) -> "EntityBuilder":
        if (
            self._value is not None
            and self._value >= threshold
            and self._state == "ACTIVE"
        ):
            self._state = "WARNING"
        return self

    def critical_above(self, threshold: float) -> "EntityBuilder":
        if self._value is not None and self._value >= threshold:
            self._state = "CRITICAL"
        return self

    def warn_below(self, threshold: float) -> "EntityBuilder":
        if (
            self._value is not None
            and self._value <= threshold
            and self._state == "ACTIVE"
        ):
            self._state = "WARNING"
        return self

    def critical_below(self, threshold: float) -> "EntityBuilder":
        if self._value is not None and self._value <= threshold:
            self._state = "CRITICAL"
        return self

    # ── Position ─────────────────────────────────────────────────────────────

    def at(self, lat: float, lon: float, alt: float = 0.0) -> "EntityBuilder":
        self._lat = lat
        self._lon = lon
        self._alt = alt
        return self

    def moving(
        self, heading: float, speed_mps: float, climb_mps: float = 0.0
    ) -> "EntityBuilder":
        self._heading = heading
        self._speed = speed_mps
        self._climb = climb_mps
        return self

    # ── Classification ───────────────────────────────────────────────────────

    def label(self, class_label: str) -> "EntityBuilder":
        self._class_label = class_label
        return self

    def confidence(self, c: float) -> "EntityBuilder":
        self._confidence = max(0.0, min(1.0, c))
        return self

    # ── Value (for sensor readings) ──────────────────────────────────────────

    def value(self, v: float, unit: str = "") -> "EntityBuilder":
        self._value = v
        self._unit = unit
        self._metadata["value"] = str(round(v, 6))
        self._metadata["unit"] = unit
        return self

    # ── Provenance ───────────────────────────────────────────────────────────

    def source(self, source_type: str, source_id: str) -> "EntityBuilder":
        self._source_type = source_type
        self._source_id = source_id
        return self

    def org(self, org_id: str) -> "EntityBuilder":
        self._org_id = org_id
        return self

    def ttl(self, seconds: int) -> "EntityBuilder":
        self._ttl = seconds
        return self

    # ── Extra metadata ───────────────────────────────────────────────────────

    def meta(self, key: str, value: str) -> "EntityBuilder":
        self._metadata[key] = value
        return self

    def meta_dict(self, d: Dict[str, str]) -> "EntityBuilder":
        self._metadata.update(d)
        return self

    # ── Aerial telemetry (drones) ────────────────────────────────────────────

    def aerial_telemetry(
        self,
        flight_mode: str = "GUIDED",
        battery_pct: float = 100.0,
        airspeed_mps: float = 0.0,
        link_quality: str = "good",
    ) -> "EntityBuilder":
        self._aerial = {
            "altitude_agl": self._alt,
            "altitude_msl": self._alt,
            "airspeed_mps": airspeed_mps,
            "flight_mode": flight_mode,
            "battery_pct": battery_pct,
            "link_quality": link_quality,
        }
        return self

    # ── Build ────────────────────────────────────────────────────────────────

    def build(self) -> Dict[str, Any]:
        now = time.time()
        now_iso = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()

        entity: Dict[str, Any] = {
            "entity_id": self._id,
            "id": self._id,
            "entity_type": self._entity_type,
            "domain": self._domain,
            "state": self._state,
            "name": self._name,
            "class_label": self._class_label,
            "confidence": self._confidence,
            "kinematics": {
                "position": {
                    "latitude": self._lat,
                    "longitude": self._lon,
                    "altitude_msl": self._alt,
                    "altitude_agl": 0.0,
                },
                "heading_deg": self._heading,
                "speed_mps": self._speed,
                "climb_rate": self._climb,
            },
            "provenance": {
                "source_id": self._source_id or self._id,
                "source_type": self._source_type,
                "org_id": self._org_id,
                "created_at": now,
                "updated_at": now,
                "version": 1,
            },
            "metadata": self._metadata,
            "ttl_seconds": self._ttl,
            "ts": now_iso,
        }

        if self._aerial is not None:
            entity["aerial"] = self._aerial

        return entity
