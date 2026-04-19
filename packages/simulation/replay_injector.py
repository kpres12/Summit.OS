"""
Replay Injector — Heli.OS

Reads mission replay data (from ReplayPersistence) and re-injects it into
the system at configurable speeds (1x, 2x, 4x). Used for regression testing,
operator training, and incident reconstruction.

Injects via MQTT summit/telemetry/{asset_id} at the original timestamps
(scaled by replay_speed).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

logger = logging.getLogger("simulation.replay_injector")


class ReplayInjector:
    """Re-injects recorded mission snapshots into the system via MQTT."""

    MIN_SPEED = 0.1
    MAX_SPEED = 10.0

    def __init__(self, mqtt_client: Any, replay_speed: float = 1.0):
        """
        Parameters
        ----------
        mqtt_client:
            Any object with a publish(topic, payload) method (e.g. paho MQTT client).
        replay_speed:
            Playback multiplier. 1.0 = real-time, 2.0 = 2x speed, etc.
        """
        self.mqtt_client = mqtt_client
        self._speed = max(self.MIN_SPEED, min(self.MAX_SPEED, replay_speed))

    # ------------------------------------------------------------------
    def set_speed(self, speed: float) -> None:
        """Set replay speed. Clamped to [0.1, 10.0]."""
        self._speed = max(self.MIN_SPEED, min(self.MAX_SPEED, speed))
        logger.info("Replay speed set to %.2fx", self._speed)

    # ------------------------------------------------------------------
    async def inject_mission(self, mission_id: str, snapshots: List[dict]) -> None:
        """
        Replay all snapshots for a mission at the configured speed.
        Snapshots are played back with inter-snapshot delays scaled by replay_speed.
        """
        if not snapshots:
            logger.warning("inject_mission: no snapshots for mission_id=%s", mission_id)
            return

        logger.info(
            "Starting replay of mission_id=%s  snapshots=%d  speed=%.2fx",
            mission_id, len(snapshots), self._speed,
        )

        prev_ts: float | None = None

        for snapshot in snapshots:
            ts_str = snapshot.get("ts_iso", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
            except Exception:
                ts = time.time()

            # Sleep to maintain relative timing
            if prev_ts is not None:
                gap = (ts - prev_ts) / self._speed
                if gap > 0:
                    await asyncio.sleep(gap)

            await self.inject_snapshot(snapshot)
            prev_ts = ts

        logger.info("Replay complete for mission_id=%s", mission_id)

    # ------------------------------------------------------------------
    async def inject_snapshot(self, snapshot: dict) -> None:
        """Publish each assignment's position to MQTT."""
        assignments = snapshot.get("assignments", [])
        ts_iso = snapshot.get("ts_iso", datetime.now(timezone.utc).isoformat())

        for assignment in assignments:
            asset_id = assignment.get("asset_id", "unknown")
            topic = f"summit/telemetry/{asset_id}"
            payload = json.dumps({
                "ts_iso":   ts_iso,
                "asset_id": asset_id,
                "lat":      assignment.get("lat"),
                "lon":      assignment.get("lon"),
                "status":   assignment.get("status"),
                "replay":   True,
            })
            try:
                self.mqtt_client.publish(topic, payload)
            except Exception as exc:
                logger.error("MQTT publish failed for asset=%s: %s", asset_id, exc)
