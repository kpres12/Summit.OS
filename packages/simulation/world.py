"""
World Simulation for Summit.OS

Provides a lightweight discrete-event simulation of:
- Entity movement (constant velocity, waypoint following)
- Sensor detection (probabilistic, range-gated)
- Environmental conditions
- Time management

Used for testing, training, and mission rehearsal without hardware.
"""
from __future__ import annotations

import math
import random
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("simulation.world")


@dataclass
class SimEntity:
    """An entity in the simulated world."""
    entity_id: str
    lat: float = 0.0
    lon: float = 0.0
    alt: float = 0.0
    heading_deg: float = 0.0
    speed_mps: float = 0.0
    entity_type: str = "unknown"  # friendly, hostile, neutral, unknown
    domain: str = "aerial"  # aerial, ground, maritime
    classification: str = ""
    # Waypoint following
    waypoints: List[Tuple[float, float, float]] = field(default_factory=list)
    current_wp_idx: int = 0
    loop_waypoints: bool = True
    # Signature
    radar_cross_section_m2: float = 1.0
    ir_signature: float = 1.0
    active: bool = True

    def distance_to(self, lat: float, lon: float) -> float:
        """Approximate distance in meters using equirectangular projection."""
        dlat = (lat - self.lat) * 111320
        dlon = (lon - self.lon) * 111320 * math.cos(math.radians(self.lat))
        return math.sqrt(dlat ** 2 + dlon ** 2)

    def bearing_to(self, lat: float, lon: float) -> float:
        """Bearing in degrees to a point."""
        dlat = (lat - self.lat) * 111320
        dlon = (lon - self.lon) * 111320 * math.cos(math.radians(self.lat))
        return math.degrees(math.atan2(dlon, dlat)) % 360


@dataclass
class SimSensor:
    """A simulated sensor."""
    sensor_id: str
    lat: float = 0.0
    lon: float = 0.0
    alt: float = 0.0
    sensor_type: str = "radar"  # radar, eo_ir, adsb, acoustic
    max_range_m: float = 10000.0
    fov_deg: float = 360.0
    boresight_deg: float = 0.0
    detection_probability: float = 0.9
    position_noise_m: float = 15.0
    update_interval_sec: float = 1.0
    attached_to: str = ""  # entity_id if mounted on a platform
    active: bool = True
    _last_scan: float = 0.0


@dataclass
class Detection:
    """A sensor detection event."""
    sensor_id: str
    entity_id: str
    lat: float
    lon: float
    alt: float
    timestamp: float
    sigma_m: float
    sensor_type: str = ""
    classification: str = ""


class WorldSimulator:
    """
    Discrete-time world simulator.

    Steps the world forward in fixed time increments,
    moving entities along waypoints and generating sensor detections.
    """

    def __init__(self, dt: float = 1.0, seed: int | None = None):
        self.dt = dt  # simulation timestep in seconds
        self.sim_time: float = 0.0
        self.entities: Dict[str, SimEntity] = {}
        self.sensors: Dict[str, SimSensor] = {}
        self._rng = random.Random(seed)
        self._callbacks: Dict[str, List[Callable]] = {}
        self._step_count: int = 0

    def add_entity(self, entity: SimEntity):
        self.entities[entity.entity_id] = entity

    def add_sensor(self, sensor: SimSensor):
        self.sensors[sensor.sensor_id] = sensor

    def on(self, event: str, callback: Callable):
        """Register callback: 'detection', 'step', 'entity_reached_wp'."""
        self._callbacks.setdefault(event, []).append(callback)

    def _emit(self, event: str, data: Any):
        for cb in self._callbacks.get(event, []):
            cb(data)

    def step(self) -> List[Detection]:
        """Advance simulation by one timestep. Returns new detections."""
        self.sim_time += self.dt
        self._step_count += 1

        # Move entities
        for ent in self.entities.values():
            if not ent.active:
                continue
            self._move_entity(ent)

        # Update sensor positions (for mounted sensors)
        for sensor in self.sensors.values():
            if sensor.attached_to and sensor.attached_to in self.entities:
                host = self.entities[sensor.attached_to]
                sensor.lat = host.lat
                sensor.lon = host.lon
                sensor.alt = host.alt

        # Generate detections
        detections = []
        for sensor in self.sensors.values():
            if not sensor.active:
                continue
            if self.sim_time - sensor._last_scan < sensor.update_interval_sec:
                continue
            sensor._last_scan = self.sim_time

            for ent in self.entities.values():
                if not ent.active:
                    continue
                det = self._try_detect(sensor, ent)
                if det:
                    detections.append(det)
                    self._emit("detection", det)

        self._emit("step", {"time": self.sim_time, "step": self._step_count,
                            "detections": len(detections)})
        return detections

    def run(self, duration_sec: float) -> List[Detection]:
        """Run simulation for a duration, returning all detections."""
        all_detections = []
        steps = int(duration_sec / self.dt)
        for _ in range(steps):
            all_detections.extend(self.step())
        return all_detections

    def _move_entity(self, ent: SimEntity):
        """Move entity along waypoints or on heading."""
        if ent.waypoints and ent.current_wp_idx < len(ent.waypoints):
            target = ent.waypoints[ent.current_wp_idx]
            dist = ent.distance_to(target[0], target[1])

            if dist < ent.speed_mps * self.dt * 2:
                # Reached waypoint
                ent.lat, ent.lon, ent.alt = target
                ent.current_wp_idx += 1
                if ent.current_wp_idx >= len(ent.waypoints) and ent.loop_waypoints:
                    ent.current_wp_idx = 0
                self._emit("entity_reached_wp", {
                    "entity_id": ent.entity_id, "wp_idx": ent.current_wp_idx - 1
                })
            else:
                # Navigate toward waypoint
                ent.heading_deg = ent.bearing_to(target[0], target[1])
                self._advance_position(ent)
        elif ent.speed_mps > 0:
            # Just fly on heading
            self._advance_position(ent)

    def _advance_position(self, ent: SimEntity):
        """Move entity forward on current heading by speed * dt."""
        dist_m = ent.speed_mps * self.dt
        heading_rad = math.radians(ent.heading_deg)
        dlat = dist_m * math.cos(heading_rad) / 111320
        dlon = dist_m * math.sin(heading_rad) / (111320 * max(0.01, math.cos(math.radians(ent.lat))))
        ent.lat += dlat
        ent.lon += dlon

    def _try_detect(self, sensor: SimSensor, entity: SimEntity) -> Optional[Detection]:
        """Try to detect an entity with a sensor. Returns Detection or None."""
        dist = math.sqrt(
            ((entity.lat - sensor.lat) * 111320) ** 2 +
            ((entity.lon - sensor.lon) * 111320 * math.cos(math.radians(sensor.lat))) ** 2 +
            (entity.alt - sensor.alt) ** 2
        )

        # Range gate
        if dist > sensor.max_range_m:
            return None

        # FOV check
        if sensor.fov_deg < 360:
            bearing = math.degrees(math.atan2(
                (entity.lon - sensor.lon) * math.cos(math.radians(sensor.lat)),
                entity.lat - sensor.lat
            )) % 360
            delta = abs(((bearing - sensor.boresight_deg + 180) % 360) - 180)
            if delta > sensor.fov_deg / 2:
                return None

        # Probabilistic detection (modified by RCS and range)
        range_factor = max(0.1, 1.0 - (dist / sensor.max_range_m) ** 2)
        rcs_factor = min(2.0, entity.radar_cross_section_m2) if sensor.sensor_type == "radar" else 1.0
        pd = sensor.detection_probability * range_factor * min(1.0, rcs_factor)

        if self._rng.random() > pd:
            return None

        # Generate noisy measurement
        noise_m = sensor.position_noise_m * (1 + dist / sensor.max_range_m)
        noise_lat = self._rng.gauss(0, noise_m / 111320)
        noise_lon = self._rng.gauss(0, noise_m / (111320 * max(0.01, math.cos(math.radians(entity.lat)))))
        noise_alt = self._rng.gauss(0, noise_m * 0.5)

        return Detection(
            sensor_id=sensor.sensor_id,
            entity_id=entity.entity_id,
            lat=entity.lat + noise_lat,
            lon=entity.lon + noise_lon,
            alt=entity.alt + noise_alt,
            timestamp=self.sim_time,
            sigma_m=noise_m,
            sensor_type=sensor.sensor_type,
            classification=entity.classification if self._rng.random() < 0.7 else "",
        )

    def get_state(self) -> Dict:
        return {
            "sim_time": self.sim_time,
            "step_count": self._step_count,
            "entities": {eid: {"lat": e.lat, "lon": e.lon, "alt": e.alt,
                               "heading": e.heading_deg, "type": e.entity_type}
                         for eid, e in self.entities.items() if e.active},
            "sensors": {sid: {"lat": s.lat, "lon": s.lon, "type": s.sensor_type,
                              "active": s.active}
                        for sid, s in self.sensors.items()},
        }
