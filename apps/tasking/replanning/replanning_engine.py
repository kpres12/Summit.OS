"""
Replanning Engine — Heli.OS Tasking

Consumes TriggerEvents from the TriggerMonitor queue and generates
updated mission assignments. Strategy per trigger type:

  ASSET_FAILED   → reassign failed asset's tasks to best available backup
  BATTERY_LOW    → dispatch RTB command to low-battery asset, reassign tasks
  THREAT_NEAR    → reroute waypoints to avoid threat, increase altitude
  WEATHER_DEGRADED → abort non-critical assets, keep critical ones at lower altitude
  OFF_COURSE     → recalculate path from current position

Publishes updated assignment via MQTT summit/missions/{mission_id}/replan
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

import httpx

from .trigger_monitor import TriggerEvent
from .replanning_state import ReplanStateStore

logger = logging.getLogger("tasking.replanning.engine")

MAX_REPLANS_PER_MISSION = int(5)  # prevents thrash


class ReplanningEngine:
    """Consumes TriggerEvents and generates updated mission plans."""

    def __init__(
        self,
        trigger_queue: asyncio.Queue,
        db_engine: Any,
        mqtt_client: Any,
        fabric_url: str,
        state_store: Optional[ReplanStateStore] = None,
    ):
        self._queue = trigger_queue
        self._db = db_engine
        self._mqtt = mqtt_client
        self._fabric_url = fabric_url.rstrip("/")
        self._state_store = state_store
        # mission_id → replan_count (in-memory guard alongside persisted state)
        self._replan_counts: dict[str, int] = {}

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Consume trigger_queue and dispatch to handler methods."""
        logger.info("ReplanningEngine started")
        while True:
            try:
                event: TriggerEvent = await self._queue.get()
                try:
                    await self._handle_trigger(event)
                except Exception as e:
                    logger.error(
                        f"ReplanningEngine error handling {event.trigger_type} "
                        f"for mission {event.mission_id}: {e}",
                        exc_info=True,
                    )
                finally:
                    self._queue.task_done()
            except asyncio.CancelledError:
                break

    # ── Dispatch ──────────────────────────────────────────────────────────────

    async def _handle_trigger(self, event: TriggerEvent) -> None:
        mission_id = event.mission_id

        # Enforce max-replan guard
        count = self._replan_counts.get(mission_id, 0)
        if count >= MAX_REPLANS_PER_MISSION:
            logger.warning(
                f"Mission {mission_id} has reached max replans "
                f"({MAX_REPLANS_PER_MISSION}), ignoring {event.trigger_type}"
            )
            return

        logger.info(
            f"Handling trigger {event.trigger_type} for mission {mission_id} "
            f"(replan #{count + 1})"
        )

        handler = {
            "ASSET_FAILED":      self._replan_asset_failed,
            "BATTERY_LOW":       self._replan_battery_low,
            "THREAT_NEAR":       self._replan_threat_near,
            "WEATHER_DEGRADED":  self._replan_weather_degraded,
            "OFF_COURSE":        self._replan_off_course,
        }.get(event.trigger_type)

        if handler is None:
            logger.warning(f"Unknown trigger type: {event.trigger_type}")
            return

        if event.trigger_type in ("ASSET_FAILED", "BATTERY_LOW", "OFF_COURSE"):
            await handler(mission_id, event.asset_id)
        elif event.trigger_type == "THREAT_NEAR":
            await handler(mission_id, event.details)
        elif event.trigger_type == "WEATHER_DEGRADED":
            await handler(mission_id, event.details)

        self._replan_counts[mission_id] = count + 1

        # Update persisted replan count
        if self._state_store:
            state = await self._state_store.load(mission_id)
            if state:
                state.replan_count = self._replan_counts[mission_id]
                state.last_replan_ts = time.time()
                await self._state_store.save(state)

    # ── Strategy implementations ──────────────────────────────────────────────

    async def _replan_asset_failed(
        self, mission_id: str, failed_asset_id: Optional[str]
    ) -> None:
        """Reassign the failed asset's tasks to the best available backup."""
        if not failed_asset_id:
            logger.warning(f"ASSET_FAILED for {mission_id} missing asset_id")
            return

        # Fetch available assets from Fabric
        available = await self._fetch_available_assets(mission_id, exclude=failed_asset_id)
        if not available:
            logger.warning(
                f"No available backup assets for mission {mission_id} "
                f"after failure of {failed_asset_id}"
            )
            return

        # Pick the closest/best asset (first returned for now — assignment engine
        # can refine this with constraint solving)
        backup_asset = available[0].get("entity_id") or available[0].get("id")
        new_assignments = {
            backup_asset: "reassigned_from_" + failed_asset_id,
        }

        await self._publish_replan(
            mission_id,
            {
                "strategy": "ASSET_FAILED",
                "failed_asset": failed_asset_id,
                "new_assignments": new_assignments,
            },
        )

    async def _replan_battery_low(
        self, mission_id: str, asset_id: Optional[str]
    ) -> None:
        """Dispatch RTB to low-battery asset and reassign its tasks."""
        if not asset_id:
            return

        # RTB command
        rtb_payload = {
            "command": "RTB",
            "asset_id": asset_id,
            "reason": "BATTERY_LOW",
            "ts": time.time(),
        }
        if self._mqtt:
            try:
                self._mqtt.publish(
                    f"summit/assets/{asset_id}/commands",
                    json.dumps(rtb_payload),
                    qos=1,
                )
            except Exception as e:
                logger.warning(f"MQTT RTB publish failed: {e}")

        # Reassign tasks
        available = await self._fetch_available_assets(mission_id, exclude=asset_id)
        new_assignments: dict = {}
        if available:
            backup = available[0].get("entity_id") or available[0].get("id")
            new_assignments[backup] = "rtb_replacement_for_" + asset_id

        await self._publish_replan(
            mission_id,
            {
                "strategy": "BATTERY_LOW",
                "rtb_asset": asset_id,
                "new_assignments": new_assignments,
            },
        )

    async def _replan_threat_near(self, mission_id: str, details: dict) -> None:
        """Reroute waypoints to avoid threat and increase altitude."""
        threat_dist = details.get("threat_proximity_m", 0)
        new_assignments = {
            "_all": {
                "action": "REROUTE_AVOID_THREAT",
                "altitude_increase_m": 30,
                "threat_proximity_m": threat_dist,
            }
        }
        await self._publish_replan(
            mission_id,
            {
                "strategy": "THREAT_NEAR",
                "threat_proximity_m": threat_dist,
                "new_assignments": new_assignments,
            },
        )

    async def _replan_weather_degraded(
        self, mission_id: str, details: dict
    ) -> None:
        """Abort non-critical assets; keep critical ones at lower altitude."""
        weather_score = details.get("weather_score", 0.0)
        new_assignments = {
            "_non_critical": {"action": "ABORT"},
            "_critical": {
                "action": "REDUCE_ALTITUDE",
                "target_alt_m": 30,
                "weather_score": weather_score,
            },
        }
        await self._publish_replan(
            mission_id,
            {
                "strategy": "WEATHER_DEGRADED",
                "weather_score": weather_score,
                "new_assignments": new_assignments,
            },
        )

    async def _replan_off_course(
        self, mission_id: str, asset_id: Optional[str]
    ) -> None:
        """Recalculate path from current position for an off-course asset."""
        if not asset_id:
            return
        entity = await self._fetch_entity(asset_id)
        pos = {}
        if entity:
            raw_pos = entity.get("position") or {}
            pos = {
                "lat": float(raw_pos.get("lat") or entity.get("latitude") or 0),
                "lon": float(raw_pos.get("lon") or entity.get("longitude") or 0),
            }

        new_assignments = {
            asset_id: {
                "action": "RECALCULATE_PATH",
                "from_position": pos,
            }
        }
        await self._publish_replan(
            mission_id,
            {
                "strategy": "OFF_COURSE",
                "asset_id": asset_id,
                "current_position": pos,
                "new_assignments": new_assignments,
            },
        )

    # ── Publishing ────────────────────────────────────────────────────────────

    async def _publish_replan(
        self, mission_id: str, new_assignments: dict
    ) -> None:
        """Publish updated mission assignments via MQTT."""
        payload = json.dumps(
            {
                "mission_id": mission_id,
                "replan_count": self._replan_counts.get(mission_id, 0),
                "ts": time.time(),
                **new_assignments,
            }
        )
        topic = f"summit/missions/{mission_id}/replan"
        if self._mqtt:
            try:
                self._mqtt.publish(topic, payload, qos=1)
                logger.info(f"Replan published to {topic}")
            except Exception as e:
                logger.warning(f"MQTT replan publish failed for {mission_id}: {e}")
        else:
            logger.debug(f"No MQTT client — replan payload: {payload}")

    # ── Fabric helpers ────────────────────────────────────────────────────────

    async def _fetch_entity(self, entity_id: str) -> Optional[dict]:
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

    async def _fetch_available_assets(
        self, mission_id: str, exclude: Optional[str] = None
    ) -> list:
        """Fetch available (non-busy) assets from Fabric."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(
                    f"{self._fabric_url}/api/v1/entities",
                    params={"type": "ASSET", "status": "AVAILABLE"},
                )
                if r.status_code == 200:
                    data = r.json()
                    assets = data.get("entities") or data.get("items") or []
                    return [
                        a for a in assets
                        if (a.get("entity_id") or a.get("id")) != exclude
                    ]
        except Exception as e:
            logger.debug(f"_fetch_available_assets failed: {e}")
        return []
