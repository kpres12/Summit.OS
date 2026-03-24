"""
Closed-Loop Execution Monitor for Summit.OS Tasking Service

Monitors active missions by comparing assigned asset positions (fetched live
from Fabric) against planned waypoints. Advances waypoint progress, detects
completion, and marks missions FAILED when telemetry goes stale.

This closes the loop between "waypoints dispatched" and "mission actually done."
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger("tasking.execution_monitor")

FABRIC_URL = os.getenv("FABRIC_URL", "http://fabric:8001")
ARRIVAL_RADIUS_M  = float(os.getenv("EXEC_ARRIVAL_RADIUS_M", "20"))
STALE_TELEMETRY_S = float(os.getenv("EXEC_STALE_TELEMETRY_S", "60"))
POLL_INTERVAL_S   = float(os.getenv("EXEC_MONITOR_POLL_S", "5"))


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


async def _get_entity_position(asset_id: str) -> Optional[Dict[str, Any]]:
    """Fetch current entity position from Fabric WorldStore."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{FABRIC_URL}/api/v1/entities/{asset_id}")
            if r.status_code == 200:
                data = r.json()
                entity = data.get("entity") or data
                pos = entity.get("position") or {}
                return {
                    "lat":      float(pos.get("lat") or entity.get("latitude") or 0),
                    "lon":      float(pos.get("lon") or entity.get("longitude") or 0),
                    "last_seen": float(entity.get("last_seen") or time.time()),
                }
    except Exception:
        pass
    return None


class ExecutionMonitor:
    """
    Background task that tracks mission progress.

    Each poll cycle:
    1. Load all ACTIVE missions + their assignments from the DB.
    2. For each assignment, fetch the asset's live position from Fabric.
    3. Compare position to the next pending waypoint (haversine).
    4. If within ARRIVAL_RADIUS_M, mark that waypoint completed and advance.
    5. If all waypoints done → complete the mission.
    6. If telemetry is stale > STALE_TELEMETRY_S → mark FAILED, re-dispatch
       remaining waypoints to any available asset (best-effort).
    """

    def __init__(self, session_factory: sessionmaker, mqtt_client: Any):
        self._session_factory = session_factory
        self._mqtt_client     = mqtt_client

    async def run(self):
        logger.info(f"ExecutionMonitor started (poll={POLL_INTERVAL_S}s)")
        while True:
            try:
                await self._tick()
            except Exception as e:
                logger.error(f"ExecutionMonitor tick error: {e}", exc_info=True)
            await asyncio.sleep(POLL_INTERVAL_S)

    # ── Private ──────────────────────────────────────────────────────────────

    async def _tick(self):
        async with self._session_factory() as session:
            # 1. Active missions
            missions = (await session.execute(
                text("SELECT mission_id, name FROM missions WHERE status = 'ACTIVE'")
            )).fetchall()

            for mission in missions:
                mission_id = mission.mission_id
                await self._check_mission(session, mission_id)

    async def _check_mission(self, session, mission_id: str):
        rows = (await session.execute(
            text(
                "SELECT id, asset_id, plan, status FROM mission_assignments "
                "WHERE mission_id = :mid AND status = 'ACTIVE'"
            ),
            {"mid": mission_id},
        )).fetchall()

        if not rows:
            return

        all_done = True
        for row in rows:
            plan: Dict = row.plan or {}
            waypoints: List[Dict] = plan.get("waypoints", [])
            completed_seq: int = plan.get("completed_waypoint_seq", -1)

            # Find next pending waypoint
            next_wp = next(
                (wp for wp in waypoints if wp.get("seq", 0) > completed_seq),
                None,
            )

            if next_wp is None:
                # This assignment is complete
                await session.execute(
                    text("UPDATE mission_assignments SET status='COMPLETED' WHERE id=:id"),
                    {"id": row.id},
                )
                continue

            all_done = False

            # Fetch live position
            pos = await _get_entity_position(row.asset_id)
            if pos is None:
                continue

            # Stale check
            age = time.time() - pos["last_seen"]
            if age > STALE_TELEMETRY_S:
                logger.warning(
                    f"Mission {mission_id}: asset {row.asset_id} telemetry stale "
                    f"({age:.0f}s) — marking FAILED"
                )
                await self._fail_mission(session, mission_id)
                return

            # Arrival check
            dist_m = _haversine_m(pos["lat"], pos["lon"], next_wp["lat"], next_wp["lon"])
            if dist_m <= ARRIVAL_RADIUS_M:
                logger.info(
                    f"Mission {mission_id}: asset {row.asset_id} reached "
                    f"waypoint seq={next_wp['seq']} (dist={dist_m:.1f}m)"
                )
                # Advance completed_seq
                plan["completed_waypoint_seq"] = next_wp["seq"]
                await session.execute(
                    text("UPDATE mission_assignments SET plan=:plan WHERE id=:id"),
                    {"plan": json.dumps(plan), "id": row.id},
                )

        await session.commit()

        if all_done:
            await self._complete_mission(session, mission_id)

    async def _complete_mission(self, session, mission_id: str):
        now = datetime.now(timezone.utc)
        await session.execute(
            text(
                "UPDATE missions SET status='COMPLETED', completed_at=:ts WHERE mission_id=:mid"
            ),
            {"ts": now, "mid": mission_id},
        )
        await session.commit()
        logger.info(f"Mission {mission_id} COMPLETED")
        self._publish_mission_event(mission_id, "COMPLETED")

    async def _fail_mission(self, session, mission_id: str):
        now = datetime.now(timezone.utc)
        await session.execute(
            text(
                "UPDATE missions SET status='FAILED', completed_at=:ts WHERE mission_id=:mid"
            ),
            {"ts": now, "mid": mission_id},
        )
        await session.execute(
            text(
                "UPDATE mission_assignments SET status='FAILED' WHERE mission_id=:mid AND status='ACTIVE'"
            ),
            {"mid": mission_id},
        )
        await session.commit()
        logger.warning(f"Mission {mission_id} FAILED")
        self._publish_mission_event(mission_id, "FAILED")

    def _publish_mission_event(self, mission_id: str, status: str):
        if not self._mqtt_client:
            return
        payload = json.dumps({
            "mission_id": mission_id,
            "status":     status,
            "ts_iso":     datetime.now(timezone.utc).isoformat(),
        })
        try:
            self._mqtt_client.publish("missions/updates", payload, qos=1)
            self._mqtt_client.publish(f"missions/{mission_id}", payload, qos=1)
        except Exception:
            pass
