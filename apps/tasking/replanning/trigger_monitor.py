"""
Trigger Monitor — Heli.OS Tasking

Watches active missions for conditions that should trigger a replan:
  - ASSET_FAILED: entity goes offline or last_seen > 60s
  - BATTERY_LOW: battery < 20%
  - THREAT_NEAR: threat entity within 500m of mission area
  - WEATHER_DEGRADED: weather score < 0.3
  - OFF_COURSE: asset deviation > 200m from planned path

Polls Fabric WorldStore every 10s. Publishes TriggerEvent to an asyncio.Queue
that the ReplanningEngine consumes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import httpx

from .replanning_state import MissionReplanState, ReplanStateStore

logger = logging.getLogger("tasking.replanning.trigger_monitor")

# ── Thresholds ────────────────────────────────────────────────────────────────
STALE_ASSET_S: float = float(os.getenv("REPLAN_STALE_ASSET_S", "60"))
BATTERY_LOW_PCT: float = float(os.getenv("REPLAN_BATTERY_LOW_PCT", "20"))
THREAT_NEAR_M: float = float(os.getenv("REPLAN_THREAT_NEAR_M", "500"))
WEATHER_DEGRADED_THRESHOLD: float = float(os.getenv("REPLAN_WEATHER_THRESHOLD", "0.3"))
OFF_COURSE_M: float = float(os.getenv("REPLAN_OFF_COURSE_M", "200"))
POLL_INTERVAL_S: float = float(os.getenv("REPLAN_POLL_INTERVAL_S", "10"))
COOLDOWN_S: float = float(os.getenv("REPLAN_COOLDOWN_S", "120"))


@dataclass
class TriggerEvent:
    """A condition detected on an active mission that warrants a replan."""

    mission_id: str
    trigger_type: str           # ASSET_FAILED / BATTERY_LOW / THREAT_NEAR / WEATHER_DEGRADED / OFF_COURSE
    asset_id: Optional[str]
    details: dict
    ts: float = field(default_factory=time.time)


class TriggerMonitor:
    """Polls active missions and emits TriggerEvents when replan conditions occur."""

    def __init__(
        self,
        trigger_queue: asyncio.Queue,
        fabric_url: str,
        state_store: ReplanStateStore,
    ):
        self._queue = trigger_queue
        self._fabric_url = fabric_url.rstrip("/")
        self._state_store = state_store
        # cooldown: (mission_id, trigger_type) → last_emitted_ts
        self._cooldowns: Dict[Tuple[str, str], float] = {}

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Poll active missions for trigger conditions every POLL_INTERVAL_S seconds."""
        logger.info(
            f"TriggerMonitor started (poll={POLL_INTERVAL_S}s, "
            f"cooldown={COOLDOWN_S}s)"
        )
        while True:
            try:
                states = await self._state_store.all_active()
                for state in states:
                    try:
                        await self._check_mission(state)
                    except Exception as e:
                        logger.error(
                            f"TriggerMonitor error on mission {state.mission_id}: {e}",
                            exc_info=True,
                        )
            except Exception as e:
                logger.error(f"TriggerMonitor poll error: {e}", exc_info=True)
            await asyncio.sleep(POLL_INTERVAL_S)

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _check_mission(self, state: MissionReplanState) -> None:
        """Evaluate all trigger conditions for a single mission."""
        # 1. Weather check
        if state.weather_score < WEATHER_DEGRADED_THRESHOLD:
            await self._emit(
                state.mission_id,
                "WEATHER_DEGRADED",
                None,
                {"weather_score": state.weather_score},
            )

        # 2. Per-asset checks
        for asset_id, role in state.asset_assignments.items():
            entity = await self._fetch_entity(asset_id)
            if entity is None:
                # Cannot fetch — treat as failed
                await self._emit(
                    state.mission_id,
                    "ASSET_FAILED",
                    asset_id,
                    {"reason": "entity_not_found", "role": role},
                )
                continue

            last_seen = float(entity.get("last_seen") or 0)
            age_s = time.time() - last_seen

            # ASSET_FAILED — stale telemetry
            if age_s > STALE_ASSET_S:
                await self._emit(
                    state.mission_id,
                    "ASSET_FAILED",
                    asset_id,
                    {"reason": "stale_telemetry", "age_s": age_s, "role": role},
                )
                continue

            # BATTERY_LOW
            battery_pct = state.battery_levels.get(asset_id)
            if battery_pct is None:
                battery_pct = float(
                    (entity.get("telemetry") or {}).get("battery_pct", 100)
                )
            if battery_pct < BATTERY_LOW_PCT:
                await self._emit(
                    state.mission_id,
                    "BATTERY_LOW",
                    asset_id,
                    {"battery_pct": battery_pct, "role": role},
                )

            # OFF_COURSE — compare current position against last stored position
            last_pos = state.last_position.get(asset_id, {})
            if last_pos:
                pos = entity.get("position") or {}
                lat = float(pos.get("lat") or entity.get("latitude") or 0)
                lon = float(pos.get("lon") or entity.get("longitude") or 0)
                planned_lat = float(last_pos.get("planned_lat") or lat)
                planned_lon = float(last_pos.get("planned_lon") or lon)
                if planned_lat and planned_lon:
                    deviation_m = _haversine_m(lat, lon, planned_lat, planned_lon)
                    if deviation_m > OFF_COURSE_M:
                        await self._emit(
                            state.mission_id,
                            "OFF_COURSE",
                            asset_id,
                            {
                                "deviation_m": deviation_m,
                                "current": {"lat": lat, "lon": lon},
                                "planned": {"lat": planned_lat, "lon": planned_lon},
                            },
                        )

        # 3. Threat proximity check
        if state.threat_proximity_m < THREAT_NEAR_M:
            await self._emit(
                state.mission_id,
                "THREAT_NEAR",
                None,
                {"threat_proximity_m": state.threat_proximity_m},
            )

    async def _fetch_entity(self, entity_id: str) -> Optional[dict]:
        """GET entity from Fabric WorldStore."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(
                    f"{self._fabric_url}/api/v1/entities/{entity_id}"
                )
                if r.status_code == 200:
                    data = r.json()
                    return data.get("entity") or data
        except Exception as e:
            logger.debug(f"_fetch_entity({entity_id}) failed: {e}")
        return None

    async def _emit(
        self,
        mission_id: str,
        trigger_type: str,
        asset_id: Optional[str],
        details: dict,
    ) -> None:
        """Put a TriggerEvent on the queue, respecting per-(mission, type) cooldown."""
        key = (mission_id, trigger_type)
        last_emitted = self._cooldowns.get(key, 0.0)
        now = time.time()
        if now - last_emitted < COOLDOWN_S:
            return

        event = TriggerEvent(
            mission_id=mission_id,
            trigger_type=trigger_type,
            asset_id=asset_id,
            details=details,
            ts=now,
        )
        self._cooldowns[key] = now
        await self._queue.put(event)
        logger.info(
            f"TriggerEvent emitted: mission={mission_id} type={trigger_type} "
            f"asset={asset_id}"
        )


# ── Utility ───────────────────────────────────────────────────────────────────

import math


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
