"""
Mission Executor — Summit.OS Edge Agent

Executes a waypoint mission on a single asset without requiring continuous
uplink. Downloads the mission plan when connected, then executes it offline.

Behaviors:
  GOTO       — navigate to waypoint lat/lon/alt
  LOITER     — hold position for duration_s
  SCAN       — activate sensors, collect data for duration_s
  RTB        — return to base (last known home position)
  RELAY      — activate mesh relay mode (Meshtastic)
  EMERGENCY  — execute emergency landing / safe stop
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("agent.mission_executor")

# Waypoint behavior constants
BEHAVIOR_GOTO = "GOTO"
BEHAVIOR_LOITER = "LOITER"
BEHAVIOR_SCAN = "SCAN"
BEHAVIOR_RTB = "RTB"
BEHAVIOR_RELAY = "RELAY"
BEHAVIOR_EMERGENCY = "EMERGENCY"

# Final status strings
STATUS_COMPLETED = "COMPLETED"
STATUS_ABORTED = "ABORTED"
STATUS_FAILED = "FAILED"

# Safety thresholds
BATTERY_RTB_THRESHOLD = 15.0   # % — auto-RTB below this
SIGNAL_LOSS_TIMEOUT = 120.0     # seconds — auto-RTB after this long without signal


class MissionExecutor:
    """
    Executes a waypoint-based mission autonomously on a single asset.

    Thread-safe read access to the LocalWorldModel keeps the executor
    aware of battery and signal state without tight coupling to the
    comms layer. The caller supplies:
      send_command_fn(asset_id, command_dict)  — async callable
      report_fn(asset_id, event_dict)          — async callable
    """

    def __init__(
        self,
        asset_id: str,
        world_model: "LocalWorldModel",  # noqa: F821
        send_command_fn: Callable,
        report_fn: Callable,
    ) -> None:
        self.asset_id = asset_id
        self._world_model = world_model
        self._send_command = send_command_fn
        self._report = report_fn

        self._mission: Optional[Dict[str, Any]] = None
        self._waypoints: List[Dict[str, Any]] = []
        self._current_wp_idx: int = 0
        self._status: str = "IDLE"
        self._abort_requested: bool = False
        self._abort_reason: str = ""
        self._started_at: Optional[float] = None
        self._signal_lost_at: Optional[float] = None

    # ── Mission Management ───────────────────────────────────

    async def load_mission(self, mission: dict) -> bool:
        """
        Validate and store the mission plan.

        Expected schema::

            {
              "mission_id": str,
              "waypoints": [
                {
                  "index": int,
                  "behavior": "GOTO"|"LOITER"|"SCAN"|"RTB"|"RELAY"|"EMERGENCY",
                  "lat": float,          # required for GOTO
                  "lon": float,          # required for GOTO
                  "alt": float,          # optional, metres AGL
                  "duration_s": float,   # required for LOITER/SCAN
                  "radius_m": float,     # arrival radius, optional
                }
              ]
            }

        Returns True if the mission is valid and was stored.
        """
        if "waypoints" not in mission or not isinstance(mission["waypoints"], list):
            logger.error("load_mission: missing or invalid 'waypoints' field")
            return False

        known_behaviors = {
            BEHAVIOR_GOTO, BEHAVIOR_LOITER, BEHAVIOR_SCAN,
            BEHAVIOR_RTB, BEHAVIOR_RELAY, BEHAVIOR_EMERGENCY,
        }
        for i, wp in enumerate(mission["waypoints"]):
            behavior = wp.get("behavior", "").upper()
            if behavior not in known_behaviors:
                logger.error(
                    "load_mission: unknown behavior '%s' at waypoint %d", behavior, i
                )
                return False
            if behavior == BEHAVIOR_GOTO and ("lat" not in wp or "lon" not in wp):
                logger.error("load_mission: GOTO waypoint %d missing lat/lon", i)
                return False

        self._mission = mission
        self._waypoints = [dict(wp) for wp in mission["waypoints"]]
        self._current_wp_idx = 0
        self._abort_requested = False
        self._status = "LOADED"
        logger.info(
            "Mission %s loaded: %d waypoints",
            mission.get("mission_id", "unknown"),
            len(self._waypoints),
        )
        return True

    # ── Execution ────────────────────────────────────────────

    async def execute(self) -> str:
        """
        Run all waypoints sequentially.

        Returns one of: COMPLETED / ABORTED / FAILED.
        Monitors battery and signal loss, auto-RTBing when safety
        thresholds are breached.
        """
        if not self._waypoints:
            logger.error("execute called with no mission loaded")
            return STATUS_FAILED

        self._status = "EXECUTING"
        self._started_at = time.time()
        self._abort_requested = False

        await self._report(
            self.asset_id,
            {"event": "mission_start", "mission_id": self._mission.get("mission_id")},
        )

        while self._current_wp_idx < len(self._waypoints):
            if self._abort_requested:
                self._status = STATUS_ABORTED
                await self._report(
                    self.asset_id,
                    {"event": "mission_aborted", "reason": self._abort_reason},
                )
                return STATUS_ABORTED

            # Safety checks before each waypoint
            safety_action = await self._safety_check()
            if safety_action == "RTB":
                logger.warning("Safety threshold breached — injecting RTB waypoint")
                await self._execute_waypoint({"behavior": BEHAVIOR_RTB})
                self._status = STATUS_ABORTED
                await self._report(
                    self.asset_id, {"event": "mission_aborted", "reason": "safety_rtb"}
                )
                return STATUS_ABORTED

            wp = self._waypoints[self._current_wp_idx]
            logger.info(
                "Executing waypoint %d/%d: behavior=%s",
                self._current_wp_idx + 1,
                len(self._waypoints),
                wp.get("behavior"),
            )

            success = await self._execute_waypoint(wp)
            if not success:
                self._status = STATUS_FAILED
                await self._report(
                    self.asset_id,
                    {
                        "event": "waypoint_failed",
                        "index": self._current_wp_idx,
                        "behavior": wp.get("behavior"),
                    },
                )
                return STATUS_FAILED

            await self._report(
                self.asset_id,
                {
                    "event": "waypoint_complete",
                    "index": self._current_wp_idx,
                    "behavior": wp.get("behavior"),
                },
            )
            self._current_wp_idx += 1

        self._status = STATUS_COMPLETED
        await self._report(
            self.asset_id,
            {
                "event": "mission_complete",
                "mission_id": self._mission.get("mission_id"),
                "elapsed_s": time.time() - (self._started_at or time.time()),
            },
        )
        logger.info("Mission completed: %s", self._mission.get("mission_id"))
        return STATUS_COMPLETED

    async def _execute_waypoint(self, wp: dict) -> bool:
        """Dispatch a single waypoint to the appropriate behavior handler."""
        behavior = wp.get("behavior", "").upper()

        handlers = {
            BEHAVIOR_GOTO: self._do_goto,
            BEHAVIOR_LOITER: self._do_loiter,
            BEHAVIOR_SCAN: self._do_scan,
            BEHAVIOR_RTB: self._do_rtb,
            BEHAVIOR_RELAY: self._do_relay,
            BEHAVIOR_EMERGENCY: self._do_emergency,
        }

        handler = handlers.get(behavior)
        if handler is None:
            logger.error("Unknown behavior: %s", behavior)
            return False

        try:
            return await handler(wp)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Waypoint handler %s raised: %s", behavior, exc, exc_info=True)
            return False

    # ── Behavior Handlers ────────────────────────────────────

    async def _do_goto(self, wp: dict) -> bool:
        """Navigate to a lat/lon/alt position."""
        cmd = {
            "command": "GOTO",
            "lat": wp["lat"],
            "lon": wp["lon"],
            "alt": wp.get("alt", 30.0),
            "radius_m": wp.get("radius_m", 10.0),
        }
        await self._send_command(self.asset_id, cmd)
        # In production, poll position from world_model until arrival.
        # Simulated as a fixed delay here.
        await asyncio.sleep(wp.get("_sim_duration_s", 1.0))
        return True

    async def _do_loiter(self, wp: dict) -> bool:
        """Hold position for duration_s."""
        duration = wp.get("duration_s", 30.0)
        await self._send_command(self.asset_id, {"command": "LOITER", "duration_s": duration})
        await asyncio.sleep(duration)
        return True

    async def _do_scan(self, wp: dict) -> bool:
        """Activate sensors and collect data for duration_s."""
        duration = wp.get("duration_s", 30.0)
        await self._send_command(
            self.asset_id,
            {"command": "SCAN", "duration_s": duration, "sensors": wp.get("sensors", ["all"])},
        )
        await asyncio.sleep(duration)
        return True

    async def _do_rtb(self, wp: dict) -> bool:
        """Navigate back to home/base."""
        asset_state = self._world_model.get(self.asset_id) or {}
        home = asset_state.get("home_position")
        if not home:
            logger.warning("RTB: no home_position in world model for %s", self.asset_id)
        await self._send_command(self.asset_id, {"command": "RTB", "home": home})
        await asyncio.sleep(wp.get("_sim_duration_s", 1.0))
        return True

    async def _do_relay(self, wp: dict) -> bool:
        """Activate Meshtastic mesh relay mode."""
        await self._send_command(self.asset_id, {"command": "RELAY", "active": True})
        await asyncio.sleep(wp.get("duration_s", 0.0))
        return True

    async def _do_emergency(self, wp: dict) -> bool:
        """Execute emergency landing / safe stop."""
        logger.warning("EMERGENCY behavior triggered on asset %s", self.asset_id)
        await self._send_command(self.asset_id, {"command": "EMERGENCY_LAND"})
        return True

    # ── Safety Monitor ───────────────────────────────────────

    async def _safety_check(self) -> Optional[str]:
        """
        Inspect world model state for safety breaches.

        Returns "RTB" if the mission should be aborted for safety,
        None otherwise.
        """
        asset_state = self._world_model.get(self.asset_id) or {}
        battery_pct: float = asset_state.get("battery_pct", 100.0)
        signal_quality: float = asset_state.get("signal_quality", 1.0)
        now = time.time()

        # Battery check
        if battery_pct < BATTERY_RTB_THRESHOLD:
            logger.warning(
                "Battery %.1f%% below RTB threshold (%.1f%%)", battery_pct, BATTERY_RTB_THRESHOLD
            )
            return "RTB"

        # Signal loss check
        if signal_quality <= 0.0:
            if self._signal_lost_at is None:
                self._signal_lost_at = now
            elif now - self._signal_lost_at > SIGNAL_LOSS_TIMEOUT:
                logger.warning("Signal lost for >%.0fs — RTB", SIGNAL_LOSS_TIMEOUT)
                return "RTB"
        else:
            self._signal_lost_at = None

        return None

    # ── Control ──────────────────────────────────────────────

    async def abort(self, reason: str = "operator_abort") -> None:
        """Request mission abort. Takes effect before the next waypoint."""
        self._abort_requested = True
        self._abort_reason = reason
        logger.info("Abort requested on %s: %s", self.asset_id, reason)

    def get_status(self) -> dict:
        """Return current execution state snapshot."""
        total = len(self._waypoints)
        progress_pct = (
            round(self._current_wp_idx / total * 100, 1) if total > 0 else 0.0
        )
        asset_state = self._world_model.get(self.asset_id) or {}
        return {
            "asset_id": self.asset_id,
            "status": self._status,
            "current_wp_index": self._current_wp_idx,
            "total_waypoints": total,
            "progress_pct": progress_pct,
            "mission_id": (self._mission or {}).get("mission_id"),
            "battery_pct": asset_state.get("battery_pct"),
            "signal_quality": asset_state.get("signal_quality"),
        }
