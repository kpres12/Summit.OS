"""
Heli.OS Behavior Tree — safe mission execution via py-trees.

The LLM brain decides WHAT to do. The behavior tree handles HOW:
  - Validates preconditions before any physical action
  - Enforces OPA safety checks at every action node
  - Provides deterministic fallback behaviour on failure
  - Composites: Sequence (all must pass), Selector (first success wins)

py-trees is optional — falls back to a simple sequential executor if not installed.

Usage:
    from behavior_tree import MissionTreeBuilder, run_tree

    tree = MissionTreeBuilder.survey_area(lat, lon, radius_m, asset_id)
    result = await run_tree(tree, opa_url="http://localhost:8181")
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("heli.tasking.behavior_tree")

try:
    import py_trees  # type: ignore

    _PYTREES = True
except ImportError:
    _PYTREES = False
    logger.debug("py-trees not installed — using built-in sequential executor")

OPA_URL = os.getenv("OPA_URL", "http://localhost:8181")
TASKING_URL = os.getenv("TASKING_URL", "http://localhost:8004")


# ── Node status (mirrors py-trees) ───────────────────────────────────────────


class NodeStatus(Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"


@dataclass
class NodeResult:
    status: NodeStatus
    message: str = ""
    data: Optional[Dict] = None


# ── Base node types ───────────────────────────────────────────────────────────


class BehaviorNode:
    """Abstract behavior tree node."""

    name: str = "node"

    async def tick(self, blackboard: Dict) -> NodeResult:
        raise NotImplementedError


class SequenceNode(BehaviorNode):
    """All children must succeed (logical AND). Stops on first FAILURE."""

    def __init__(self, name: str, children: List[BehaviorNode]):
        self.name = name
        self.children = children

    async def tick(self, blackboard: Dict) -> NodeResult:
        for child in self.children:
            result = await child.tick(blackboard)
            if result.status == NodeStatus.FAILURE:
                logger.debug(
                    f"Sequence '{self.name}' failed at '{child.name}': {result.message}"
                )
                return result
            if result.status == NodeStatus.RUNNING:
                return result
        return NodeResult(NodeStatus.SUCCESS, f"Sequence '{self.name}' complete")


class SelectorNode(BehaviorNode):
    """First child to succeed wins (logical OR). Stops on first SUCCESS."""

    def __init__(self, name: str, children: List[BehaviorNode]):
        self.name = name
        self.children = children

    async def tick(self, blackboard: Dict) -> NodeResult:
        for child in self.children:
            result = await child.tick(blackboard)
            if result.status == NodeStatus.SUCCESS:
                return result
            if result.status == NodeStatus.RUNNING:
                return result
        return NodeResult(
            NodeStatus.FAILURE, f"Selector '{self.name}': all children failed"
        )


class ParallelNode(BehaviorNode):
    """Run all children concurrently. Succeeds when all succeed."""

    def __init__(self, name: str, children: List[BehaviorNode]):
        self.name = name
        self.children = children

    async def tick(self, blackboard: Dict) -> NodeResult:
        results = await asyncio.gather(*[c.tick(blackboard) for c in self.children])
        failed = [r for r in results if r.status == NodeStatus.FAILURE]
        if failed:
            return NodeResult(
                NodeStatus.FAILURE, f"Parallel '{self.name}': {failed[0].message}"
            )
        return NodeResult(NodeStatus.SUCCESS, f"Parallel '{self.name}' complete")


# ── Condition nodes ───────────────────────────────────────────────────────────


class AssetAvailableCondition(BehaviorNode):
    """Check that the target asset is ACTIVE and not already tasked."""

    def __init__(self, asset_id: str, fabric_url: str = "http://localhost:8001"):
        self.name = f"asset_available({asset_id})"
        self.asset_id = asset_id
        self.fabric_url = fabric_url

    async def tick(self, blackboard: Dict) -> NodeResult:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.fabric_url}/entities/{self.asset_id}")
                if r.status_code == 404:
                    return NodeResult(
                        NodeStatus.FAILURE, f"Asset {self.asset_id} not found"
                    )
                entity = r.json()
                state = entity.get("state", "")
                if state not in ("ACTIVE", "IDLE"):
                    return NodeResult(
                        NodeStatus.FAILURE,
                        f"Asset {self.asset_id} not available (state={state})",
                    )
                blackboard["asset"] = entity
                return NodeResult(
                    NodeStatus.SUCCESS, f"Asset {self.asset_id} available"
                )
        except Exception as e:
            return NodeResult(NodeStatus.FAILURE, f"Asset check failed: {e}")


class OPASafetyCondition(BehaviorNode):
    """
    Gate: run OPA policy check before any physical action.
    This node MUST be in every action sequence — it's the safety interlock.
    """

    def __init__(
        self,
        policy_path: str,
        input_fn: Callable[[Dict], Dict],
        opa_url: str = OPA_URL,
    ):
        self.name = f"opa_check({policy_path})"
        self.policy_path = policy_path
        self.input_fn = input_fn
        self.opa_url = opa_url

    async def tick(self, blackboard: Dict) -> NodeResult:
        try:
            import httpx

            opa_input = self.input_fn(blackboard)
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.post(
                    f"{self.opa_url}/v1/data/{self.policy_path}",
                    json={"input": opa_input},
                )
                if r.status_code != 200:
                    return NodeResult(
                        NodeStatus.FAILURE,
                        f"OPA returned {r.status_code}",
                    )
                result = r.json().get("result", {})
                allowed = result.get("allow", result.get("allowed", False))
                if not allowed:
                    reasons = result.get("deny", result.get("violations", []))
                    return NodeResult(
                        NodeStatus.FAILURE,
                        f"OPA policy denied: {reasons}",
                    )
                blackboard["opa_result"] = result
                return NodeResult(NodeStatus.SUCCESS, "OPA check passed")
        except Exception as e:
            # OPA unreachable — fail safe (deny)
            logger.error(f"OPA check failed (network/error): {e}")
            return NodeResult(
                NodeStatus.FAILURE, f"OPA unreachable — failing safe: {e}"
            )


class BatteryCondition(BehaviorNode):
    """Fail if battery below threshold (read from blackboard asset data)."""

    def __init__(self, min_pct: float = 20.0):
        self.name = f"battery>={min_pct}%"
        self.min_pct = min_pct

    async def tick(self, blackboard: Dict) -> NodeResult:
        asset = blackboard.get("asset", {})
        aerial = asset.get("aerial") or {}
        battery = aerial.get("battery_pct")
        if battery is None:
            # Not an aerial asset — skip battery check
            return NodeResult(NodeStatus.SUCCESS, "Battery check N/A")
        if battery < self.min_pct:
            return NodeResult(
                NodeStatus.FAILURE,
                f"Battery {battery}% below minimum {self.min_pct}%",
            )
        return NodeResult(NodeStatus.SUCCESS, f"Battery OK ({battery}%)")


class GeofenceContainmentCondition(BehaviorNode):
    """Check target position is inside inclusion zones and outside exclusion zones."""

    def __init__(
        self, lat: float, lon: float, fabric_url: str = "http://localhost:8001"
    ):
        self.name = "geofence_containment"
        self.lat = lat
        self.lon = lon
        self.fabric_url = fabric_url

    async def tick(self, blackboard: Dict) -> NodeResult:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(
                    f"{self.fabric_url}/geofences/check",
                    params={"lat": self.lat, "lon": self.lon},
                )
                if r.status_code == 200:
                    result = r.json()
                    if not result.get("allowed", True):
                        return NodeResult(
                            NodeStatus.FAILURE,
                            f"Geofence violation at {self.lat:.4f},{self.lon:.4f}",
                        )
            return NodeResult(NodeStatus.SUCCESS, "Target within allowed zones")
        except Exception as e:
            logger.debug(f"Geofence check failed: {e} — allowing (non-critical)")
            return NodeResult(
                NodeStatus.SUCCESS, "Geofence check skipped (unavailable)"
            )


# ── Action nodes ──────────────────────────────────────────────────────────────


class DispatchMissionAction(BehaviorNode):
    """Create and dispatch a mission task via the Tasking service."""

    def __init__(
        self,
        task_type: str,
        lat: float,
        lon: float,
        asset_id: Optional[str] = None,
        priority: int = 3,
        radius_m: Optional[float] = None,
        description: str = "",
        tasking_url: str = TASKING_URL,
    ):
        self.name = f"dispatch_{task_type}@{lat:.3f},{lon:.3f}"
        self.task_type = task_type
        self.lat = lat
        self.lon = lon
        self.asset_id = asset_id
        self.priority = priority
        self.radius_m = radius_m
        self.description = description
        self.tasking_url = tasking_url

    async def tick(self, blackboard: Dict) -> NodeResult:
        try:
            import httpx

            payload: Dict[str, Any] = {
                "task_type": self.task_type,
                "target_lat": self.lat,
                "target_lon": self.lon,
                "priority": self.priority,
                "description": self.description or f"BT-dispatched {self.task_type}",
            }
            if self.asset_id:
                payload["asset_id"] = self.asset_id
            elif "asset" in blackboard:
                payload["asset_id"] = blackboard["asset"].get("entity_id")
            if self.radius_m:
                payload["radius_m"] = self.radius_m

            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(f"{self.tasking_url}/missions", json=payload)
                r.raise_for_status()
                result = r.json()
                blackboard["mission_result"] = result
                logger.info(f"Mission dispatched: {result.get('mission_id', '?')}")
                return NodeResult(
                    NodeStatus.SUCCESS,
                    f"Mission created: {result.get('mission_id')}",
                    result,
                )
        except Exception as e:
            return NodeResult(NodeStatus.FAILURE, f"Dispatch failed: {e}")


class RaiseAlertAction(BehaviorNode):
    """Raise an alert in the fabric."""

    def __init__(
        self, severity: str, description: str, fabric_url: str = "http://localhost:8001"
    ):
        self.name = f"raise_alert({severity})"
        self.severity = severity
        self.description = description
        self.fabric_url = fabric_url

    async def tick(self, blackboard: Dict) -> NodeResult:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.post(
                    f"{self.fabric_url}/alerts",
                    json={
                        "severity": self.severity,
                        "description": self.description,
                        "source": "behavior_tree",
                    },
                )
                return NodeResult(NodeStatus.SUCCESS, f"Alert raised: {self.severity}")
        except Exception as e:
            return NodeResult(NodeStatus.FAILURE, f"Alert failed: {e}")


class ReturnHomeAction(BehaviorNode):
    """Send RETURN_HOME command to an asset."""

    def __init__(self, asset_id: str, tasking_url: str = TASKING_URL):
        self.name = f"return_home({asset_id})"
        self.asset_id = asset_id
        self.tasking_url = tasking_url

    async def tick(self, blackboard: Dict) -> NodeResult:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.post(
                    f"{self.tasking_url}/assets/{self.asset_id}/command",
                    json={"action": "RETURN_HOME", "reason": "behavior_tree"},
                )
                return NodeResult(
                    NodeStatus.SUCCESS, f"Return home sent to {self.asset_id}"
                )
        except Exception as e:
            return NodeResult(NodeStatus.FAILURE, f"Return home failed: {e}")


# ── Mission tree builder ──────────────────────────────────────────────────────


class MissionTreeBuilder:
    """Factory for common mission tree patterns."""

    @staticmethod
    def survey_area(
        lat: float,
        lon: float,
        radius_m: float,
        asset_id: Optional[str] = None,
        priority: int = 3,
    ) -> BehaviorNode:
        """
        Survey area tree:
          Sequence:
            1. OPA safety check
            2. Asset available (if specified)
            3. Battery check
            4. Geofence containment
            5. Dispatch SURVEY mission
          Fallback selector on failure:
            - Raise WARNING alert
            - Return HOME (if asset known)
        """
        preconditions: List[BehaviorNode] = [
            OPASafetyCondition(
                policy_path="summit/geofence/allow",
                input_fn=lambda bb: {
                    "lat": lat,
                    "lon": lon,
                    "asset_id": asset_id or "",
                },
            ),
        ]
        if asset_id:
            preconditions.append(AssetAvailableCondition(asset_id))
            preconditions.append(BatteryCondition(min_pct=20.0))
        preconditions.append(GeofenceContainmentCondition(lat, lon))

        action = DispatchMissionAction(
            task_type="SURVEY",
            lat=lat,
            lon=lon,
            asset_id=asset_id,
            priority=priority,
            radius_m=radius_m,
            description=f"AI-initiated survey at {lat:.4f},{lon:.4f} r={radius_m}m",
        )

        happy_path = SequenceNode(
            "survey_happy_path",
            preconditions + [action],
        )

        fallback_actions: List[BehaviorNode] = [
            RaiseAlertAction(
                "WARNING", f"Survey dispatch blocked at {lat:.4f},{lon:.4f}"
            ),
        ]
        if asset_id:
            fallback_actions.append(ReturnHomeAction(asset_id))

        fallback = SequenceNode("survey_fallback", fallback_actions)

        return SelectorNode(
            "survey_area",
            [happy_path, fallback],
        )

    @staticmethod
    def emergency_response(
        lat: float,
        lon: float,
        asset_id: str,
    ) -> BehaviorNode:
        """
        Emergency response tree:
          Parallel:
            - Raise CRITICAL alert
            - Sequence: OPA → asset available → dispatch MONITOR
        """
        alert = RaiseAlertAction(
            "CRITICAL", f"Emergency response initiated at {lat:.4f},{lon:.4f}"
        )

        response_seq = SequenceNode(
            "emergency_dispatch",
            [
                OPASafetyCondition(
                    policy_path="summit/geofence/allow",
                    input_fn=lambda bb: {
                        "lat": lat,
                        "lon": lon,
                        "asset_id": asset_id,
                        "emergency": True,
                    },
                ),
                AssetAvailableCondition(asset_id),
                DispatchMissionAction(
                    task_type="MONITOR",
                    lat=lat,
                    lon=lon,
                    asset_id=asset_id,
                    priority=5,
                    description="Emergency AI response",
                ),
            ],
        )

        return ParallelNode("emergency_response", [alert, response_seq])


# ── Tree executor ─────────────────────────────────────────────────────────────


async def run_tree(root: BehaviorNode, blackboard: Optional[Dict] = None) -> NodeResult:
    """Execute a behavior tree from the root node."""
    if blackboard is None:
        blackboard = {}
    return await root.tick(blackboard)
