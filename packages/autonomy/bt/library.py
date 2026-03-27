"""
Pre-built Behavior Tree Library for Summit.OS

Ready-to-use mission behavior trees for common operational patterns.
Each factory function returns a configured BehaviorTree.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from packages.autonomy.bt.nodes import (
    BTNode,
    Blackboard,
    NodeStatus,
    Sequence,
    Selector,
    Parallel,
    Inverter,
    Repeat,
    RetryUntilSuccess,
    Timeout,
    Action,
    Condition,
    Wait,
    SetBlackboard,
    CheckBlackboard,
)
from packages.autonomy.bt.tree import BehaviorTree


# ═══════════════════════════════════════════════════════════
# Common Action/Condition Factories
# ═══════════════════════════════════════════════════════════


def _check_battery(bb: Blackboard) -> bool:
    """Check if battery is above minimum threshold."""
    level = bb.get("battery_percent", 100)
    threshold = bb.get("battery_min_percent", 20)
    return level > threshold


def _check_comms(bb: Blackboard) -> bool:
    """Check if communications are available."""
    return bb.get("comms_status", "connected") != "lost"


def _check_threats(bb: Blackboard) -> bool:
    """Check if threats are detected."""
    threats = bb.get("detected_threats", [])
    return len(threats) > 0


def _navigate_to_waypoint(bb: Blackboard) -> NodeStatus:
    """Navigate to the current target waypoint."""
    waypoints = bb.get("waypoints", [])
    wp_idx = bb.get("current_waypoint_idx", 0)

    if not waypoints or wp_idx >= len(waypoints):
        return NodeStatus.SUCCESS  # All waypoints visited

    target = waypoints[wp_idx]
    current_lat = bb.get("vehicle_lat", 0.0)
    current_lon = bb.get("vehicle_lon", 0.0)

    # Simple distance check (meters, approximate)
    dlat = (target["lat"] - current_lat) * 111320
    dlon = (target["lon"] - current_lon) * 111320 * math.cos(math.radians(current_lat))
    dist = math.sqrt(dlat**2 + dlon**2)

    if dist < target.get("radius_m", 10.0):
        bb.set("current_waypoint_idx", wp_idx + 1)
        return NodeStatus.SUCCESS

    # Still navigating
    bb.set("nav_target", target)
    return NodeStatus.RUNNING


def _loiter_at_position(bb: Blackboard) -> NodeStatus:
    """Loiter at current position for specified duration."""
    import time

    loiter_start = bb.get("loiter_start_time")
    loiter_duration = bb.get("loiter_duration_sec", 60.0)

    if loiter_start is None:
        bb.set("loiter_start_time", time.time())
        return NodeStatus.RUNNING

    if time.time() - loiter_start >= loiter_duration:
        bb.delete("loiter_start_time")
        return NodeStatus.SUCCESS

    return NodeStatus.RUNNING


def _return_to_base(bb: Blackboard) -> NodeStatus:
    """Navigate back to home/base position."""
    home = bb.get("home_position")
    if not home:
        return NodeStatus.FAILURE

    bb.set("waypoints", [home])
    bb.set("current_waypoint_idx", 0)
    return _navigate_to_waypoint(bb)


def _report_position(bb: Blackboard) -> NodeStatus:
    """Report current position to C2."""
    # In real impl, this would publish to MQTT/gRPC
    bb.set(
        "last_position_report",
        {
            "lat": bb.get("vehicle_lat", 0),
            "lon": bb.get("vehicle_lon", 0),
            "alt": bb.get("vehicle_alt", 0),
        },
    )
    return NodeStatus.SUCCESS


def _scan_area(bb: Blackboard) -> NodeStatus:
    """Activate sensors to scan the area."""
    bb.set("sensors_active", True)
    bb.set("scan_mode", "wide")
    return NodeStatus.SUCCESS


def _track_target(bb: Blackboard) -> NodeStatus:
    """Switch to tracking mode for detected target."""
    threats = bb.get("detected_threats", [])
    if not threats:
        return NodeStatus.FAILURE
    bb.set("tracking_target", threats[0])
    bb.set("scan_mode", "narrow")
    return NodeStatus.SUCCESS


# ═══════════════════════════════════════════════════════════
# Pre-built Mission Trees
# ═══════════════════════════════════════════════════════════


def build_patrol_tree(waypoints: List[Dict]) -> BehaviorTree:
    """
    Patrol Mission: Visit waypoints in order, repeat.

    Structure:
      Selector(PatrolMission)
      ├── Sequence(SafetyCheck)
      │   ├── Condition(BatteryOK)
      │   └── Sequence(Patrol)
      │       ├── Repeat(VisitWaypoints)
      │       │   └── Action(NavigateToWaypoint)
      │       └── Action(ReportPosition)
      └── Sequence(RTB)
          └── Action(ReturnToBase)
    """
    root = Selector(
        "PatrolMission",
        children=[
            Sequence(
                "NormalOps",
                children=[
                    Condition("BatteryOK", predicate=_check_battery),
                    Sequence(
                        "Patrol",
                        children=[
                            Repeat(
                                "VisitWaypoints",
                                count=len(waypoints),
                                child=Action(
                                    "NavigateToWaypoint", action=_navigate_to_waypoint
                                ),
                            ),
                            Action("ReportPosition", action=_report_position),
                        ],
                    ),
                ],
            ),
            Sequence(
                "RTB",
                children=[
                    Action("ReturnToBase", action=_return_to_base),
                ],
            ),
        ],
    )

    tree = BehaviorTree(root, name="PatrolMission")
    tree.setup({"waypoints": waypoints, "current_waypoint_idx": 0})
    return tree


def build_surveillance_tree(area_center: Dict, loiter_sec: float = 300) -> BehaviorTree:
    """
    Surveillance Mission: Navigate to area, loiter while scanning.

    Structure:
      Selector(Surveillance)
      ├── Sequence(Mission)
      │   ├── Condition(BatteryOK)
      │   ├── Action(NavigateToArea)
      │   ├── Parallel(ScanAndLoiter)
      │   │   ├── Action(ScanArea)
      │   │   └── Action(Loiter)
      │   └── Selector(ThreatResponse)
      │       ├── Sequence(TrackIfThreat)
      │       │   ├── Condition(ThreatDetected)
      │       │   └── Action(TrackTarget)
      │       └── Action(ContinueScan)
      └── Action(RTB)
    """
    root = Selector(
        "Surveillance",
        children=[
            Sequence(
                "Mission",
                children=[
                    Condition("BatteryOK", predicate=_check_battery),
                    Action("NavigateToArea", action=_navigate_to_waypoint),
                    Parallel(
                        "ScanAndLoiter",
                        success_policy="all",
                        failure_policy="one",
                        children=[
                            Action("ScanArea", action=_scan_area),
                            Action("Loiter", action=_loiter_at_position),
                        ],
                    ),
                    Selector(
                        "ThreatResponse",
                        children=[
                            Sequence(
                                "TrackIfThreat",
                                children=[
                                    Condition(
                                        "ThreatDetected", predicate=_check_threats
                                    ),
                                    Action("TrackTarget", action=_track_target),
                                ],
                            ),
                            Action("ContinueScan", action=_scan_area),
                        ],
                    ),
                ],
            ),
            Action("RTB", action=_return_to_base),
        ],
    )

    tree = BehaviorTree(root, name="SurveillanceMission")
    tree.setup(
        {
            "waypoints": [area_center],
            "current_waypoint_idx": 0,
            "loiter_duration_sec": loiter_sec,
        }
    )
    return tree


def build_search_and_track_tree(search_waypoints: List[Dict]) -> BehaviorTree:
    """
    Search & Track: Patrol area looking for targets, switch to tracking when found.

    Structure:
      Selector(SearchAndTrack)
      ├── Sequence(TrackMode)
      │   ├── Condition(ThreatDetected)
      │   ├── Action(TrackTarget)
      │   └── Action(ReportPosition)
      ├── Sequence(SearchMode)
      │   ├── Condition(BatteryOK)
      │   ├── Action(ScanArea)
      │   └── Action(NavigateToWaypoint)
      └── Action(RTB)
    """
    root = Selector(
        "SearchAndTrack",
        children=[
            Sequence(
                "TrackMode",
                children=[
                    Condition("ThreatDetected", predicate=_check_threats),
                    Action("TrackTarget", action=_track_target),
                    Action("ReportPosition", action=_report_position),
                ],
            ),
            Sequence(
                "SearchMode",
                children=[
                    Condition("BatteryOK", predicate=_check_battery),
                    Action("ScanArea", action=_scan_area),
                    Action("Navigate", action=_navigate_to_waypoint),
                ],
            ),
            Action("RTB", action=_return_to_base),
        ],
    )

    tree = BehaviorTree(root, name="SearchAndTrackMission")
    tree.setup(
        {
            "waypoints": search_waypoints,
            "current_waypoint_idx": 0,
        }
    )
    return tree


def build_rtb_tree() -> BehaviorTree:
    """Simple Return to Base tree."""
    root = Sequence(
        "RTBSequence",
        children=[
            Action("ReturnToBase", action=_return_to_base),
            Action("ReportPosition", action=_report_position),
        ],
    )
    tree = BehaviorTree(root, name="ReturnToBase")
    tree.setup()
    return tree
