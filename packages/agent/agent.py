"""
Summit.OS Edge Agent

Standalone process that runs on-asset (Raspberry Pi, Jetson Nano, etc.).
Connects to the central Fabric when available, falls back to autonomous
operation when offline. Implements the execute-then-report loop.

Startup sequence:
  1. Connect to Fabric (or start in offline mode)
  2. Download pending missions
  3. Start local world model sync
  4. Execute missions
  5. Upload telemetry + completed missions on reconnect
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx

from packages.agent.local_world_model import LocalWorldModel
from packages.agent.mission_executor import MissionExecutor

logger = logging.getLogger("agent.edge")


class EdgeAgent:
    """
    Top-level edge agent process.

    Coordinates the local world model, mission executor, heartbeat,
    telemetry, and sync loops. Designed to run continuously on-asset
    and tolerate intermittent Fabric connectivity.
    """

    def __init__(
        self,
        asset_id: str,
        fabric_url: str,
        config: Dict[str, Any] = None,
    ) -> None:
        self.asset_id = asset_id
        self.fabric_url = fabric_url.rstrip("/")
        self.config = config or {}

        self._running = False
        self._online = False
        self._started_at: Optional[float] = None
        self._last_heartbeat_ok: Optional[float] = None
        self._last_telemetry_ok: Optional[float] = None
        self._last_sync_ok: Optional[float] = None
        self._active_mission_id: Optional[str] = None

        # Sub-systems
        self.world_model = LocalWorldModel(
            asset_id=asset_id,
            max_entities=self.config.get("max_entities", 200),
        )
        self._executor = MissionExecutor(
            asset_id=asset_id,
            world_model=self.world_model,
            send_command_fn=self._send_command,
            report_fn=self._report_event,
        )

        # Background task handles
        self._tasks: List[asyncio.Task] = []

    # ── Lifecycle ────────────────────────────────────────────

    async def start(self) -> None:
        """
        Main agent loop: attempt Fabric connect, then run all background loops.

        Blocks until stop() is called.
        """
        if self._running:
            return

        self._running = True
        self._started_at = time.time()
        logger.info("EdgeAgent %s starting (fabric=%s)", self.asset_id, self.fabric_url)

        # Attempt initial connection and mission download
        self._online = await self._try_connect()
        if self._online:
            await self._download_missions()
            await self.world_model.pull_from_fabric(self.fabric_url)

        # Launch background loops
        self._tasks = [
            asyncio.create_task(self._heartbeat_loop(), name="heartbeat"),
            asyncio.create_task(self._telemetry_loop(), name="telemetry"),
            asyncio.create_task(self._mission_poll_loop(), name="mission_poll"),
            asyncio.create_task(self._sync_loop(), name="sync"),
        ]

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass
        finally:
            logger.info("EdgeAgent %s stopped", self.asset_id)

    async def stop(self) -> None:
        """Gracefully stop all background loops."""
        self._running = False
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        logger.info("EdgeAgent %s shutdown complete", self.asset_id)

    # ── Background Loops ─────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        """POST heartbeat to Fabric every 10 seconds."""
        while self._running:
            try:
                url = f"{self.fabric_url}/api/v1/assets/{self.asset_id}/heartbeat"
                payload = {
                    "ts": time.time(),
                    "status": self._executor.get_status()["status"],
                    "online": self._online,
                }
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                self._online = True
                self._last_heartbeat_ok = time.time()
            except httpx.HTTPError as exc:
                if self._online:
                    logger.warning("Heartbeat failed — going offline: %s", exc)
                self._online = False
            except Exception as exc:
                logger.error("Heartbeat loop error: %s", exc, exc_info=True)
            await asyncio.sleep(10.0)

    async def _telemetry_loop(self) -> None:
        """Push position/battery/status telemetry to Fabric every 5 seconds."""
        while self._running:
            try:
                asset_state = self.world_model.get(self.asset_id) or {}
                payload = {
                    "asset_id": self.asset_id,
                    "ts": time.time(),
                    "lat": asset_state.get("lat"),
                    "lon": asset_state.get("lon"),
                    "alt": asset_state.get("alt"),
                    "battery_pct": asset_state.get("battery_pct"),
                    "signal_quality": asset_state.get("signal_quality"),
                    "executor": self._executor.get_status(),
                }
                if self._online:
                    url = f"{self.fabric_url}/api/v1/assets/{self.asset_id}/telemetry"
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        resp = await client.post(url, json=payload)
                        resp.raise_for_status()
                    self._last_telemetry_ok = time.time()
            except httpx.HTTPError as exc:
                logger.debug("Telemetry push failed: %s", exc)
                self._online = False
            except Exception as exc:
                logger.error("Telemetry loop error: %s", exc, exc_info=True)
            await asyncio.sleep(5.0)

    async def _mission_poll_loop(self) -> None:
        """Check Fabric for new pending missions every 30 seconds."""
        while self._running:
            if self._online:
                try:
                    await self._download_missions()
                except Exception as exc:
                    logger.error("Mission poll error: %s", exc, exc_info=True)
            await asyncio.sleep(30.0)

    async def _sync_loop(self) -> None:
        """Sync world model delta to Fabric every 60 seconds."""
        while self._running:
            if self._online:
                try:
                    pushed = await self.world_model.push_to_fabric(self.fabric_url)
                    pulled = await self.world_model.pull_from_fabric(self.fabric_url)
                    self._last_sync_ok = time.time()
                    logger.debug("Sync: pushed=%d pulled=%d", pushed, pulled)
                except Exception as exc:
                    logger.error("Sync loop error: %s", exc, exc_info=True)
            await asyncio.sleep(60.0)

    # ── Helpers ──────────────────────────────────────────────

    async def _try_connect(self) -> bool:
        """Probe the Fabric health endpoint. Returns True if reachable."""
        try:
            url = f"{self.fabric_url}/api/v1/health"
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
            logger.info("Fabric reachable at %s", self.fabric_url)
            return True
        except httpx.HTTPError as exc:
            logger.warning("Fabric not reachable — starting offline: %s", exc)
            return False

    async def _download_missions(self) -> None:
        """Fetch pending missions from the Fabric and queue them for execution."""
        url = f"{self.fabric_url}/api/v1/assets/{self.asset_id}/missions"
        params = {"status": "pending"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
            missions: List[dict] = resp.json().get("missions", [])
        except httpx.HTTPError as exc:
            logger.warning("Mission download failed: %s", exc)
            return

        for mission in missions:
            mid = mission.get("mission_id", "unknown")
            if mid == self._active_mission_id:
                continue
            ok = await self._executor.load_mission(mission)
            if ok:
                logger.info("Loaded mission %s — starting execution", mid)
                self._active_mission_id = mid
                # Fire-and-forget: execute in background so loops keep running.
                asyncio.create_task(
                    self._run_mission(), name=f"mission_{mid}"
                )
                break  # One mission at a time

    async def _run_mission(self) -> None:
        """Execute the loaded mission and report the result."""
        final_status = await self._executor.execute()
        logger.info("Mission %s ended: %s", self._active_mission_id, final_status)
        if self._online:
            try:
                url = (
                    f"{self.fabric_url}/api/v1/assets/{self.asset_id}"
                    f"/missions/{self._active_mission_id}/result"
                )
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.post(
                        url,
                        json={
                            "status": final_status,
                            "ts": time.time(),
                            "executor": self._executor.get_status(),
                        },
                    )
            except httpx.HTTPError as exc:
                logger.warning("Failed to report mission result: %s", exc)

    async def _send_command(self, asset_id: str, command: dict) -> None:
        """
        Forward a command to the vehicle HAL.

        In production this would publish to the MAVLink / ROS2 bridge.
        Here we log and optionally POST to the Fabric command endpoint.
        """
        logger.info("CMD %s → %s", asset_id, command.get("command"))
        if self._online:
            try:
                url = f"{self.fabric_url}/api/v1/assets/{asset_id}/command"
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.post(url, json=command)
            except httpx.HTTPError as exc:
                logger.debug("Command relay failed (offline): %s", exc)

    async def _report_event(self, asset_id: str, event: dict) -> None:
        """Emit a mission event to the Fabric events endpoint."""
        logger.info("EVENT %s: %s", asset_id, event.get("event"))
        if self._online:
            try:
                url = f"{self.fabric_url}/api/v1/assets/{asset_id}/events"
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.post(url, json={**event, "ts": time.time()})
            except httpx.HTTPError as exc:
                logger.debug("Event report failed (offline): %s", exc)

    # ── Status ───────────────────────────────────────────────

    def get_status(self) -> dict:
        """Return a snapshot of agent health and execution state."""
        return {
            "asset_id": self.asset_id,
            "fabric_url": self.fabric_url,
            "online": self._online,
            "uptime_s": (
                round(time.time() - self._started_at, 1)
                if self._started_at else None
            ),
            "last_heartbeat_ok": self._last_heartbeat_ok,
            "last_telemetry_ok": self._last_telemetry_ok,
            "last_sync_ok": self._last_sync_ok,
            "world_model_entities": len(self.world_model.all_entities()),
            "executor": self._executor.get_status(),
        }


# ── Entry Point ──────────────────────────────────────────────


async def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    asset_id = os.environ.get("ASSET_ID", "edge-asset-001")
    fabric_url = os.environ.get("FABRIC_URL", "http://fabric:8001")

    agent = EdgeAgent(asset_id=asset_id, fabric_url=fabric_url)

    try:
        await agent.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
