"""
Tests for the deterministic MissionPlanner (no LLM required).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_smoke_dispatches_survey():
    from mission_planner import MissionPlanner
    p = MissionPlanner()
    plan = p.plan({"class": "smoke", "confidence": 0.92, "lat": 34.05, "lon": -118.24})
    assert plan is not None
    assert plan.mission_type == "SURVEY"
    assert plan.loiter is True
    assert plan.priority == "CRITICAL"
    assert plan.asset_class == "UAV"


def test_fire_dispatches_survey():
    from mission_planner import MissionPlanner
    p = MissionPlanner()
    plan = p.plan({"class": "wildfire", "confidence": 0.80, "lat": 34.05, "lon": -118.24})
    assert plan is not None
    assert plan.mission_type == "SURVEY"


def test_person_dispatches_monitor():
    from mission_planner import MissionPlanner
    p = MissionPlanner()
    plan = p.plan({"class": "person", "confidence": 0.85, "lat": 34.0, "lon": -118.0})
    assert plan is not None
    assert plan.mission_type == "MONITOR"
    assert plan.loiter is True


def test_collapse_dispatches_search():
    from mission_planner import MissionPlanner
    p = MissionPlanner()
    plan = p.plan({"class": "building collapse", "confidence": 0.75, "lat": 34.0, "lon": -118.0})
    assert plan is not None
    assert plan.mission_type == "SEARCH"


def test_hazmat_dispatches_perimeter():
    from mission_planner import MissionPlanner
    p = MissionPlanner()
    plan = p.plan({"class": "chemical spill", "confidence": 0.70, "lat": 34.0, "lon": -118.0})
    assert plan is not None
    assert plan.mission_type in ("PERIMETER", "MONITOR")  # both defensible for hazmat containment


def test_low_confidence_unknown_returns_none():
    """Unknown class + low confidence should not auto-dispatch."""
    from mission_planner import MissionPlanner
    p = MissionPlanner()
    plan = p.plan({"class": "unknown_thing", "confidence": 0.50, "lat": 34.0, "lon": -118.0})
    assert plan is None


def test_high_confidence_unknown_dispatches():
    """Unknown class at very high confidence → some mission is returned (not None)."""
    from mission_planner import MissionPlanner
    p = MissionPlanner()
    plan = p.plan({"class": "alien_object", "confidence": 0.92, "lat": 34.0, "lon": -118.0})
    assert plan is not None
    assert plan.mission_type in ("SURVEY", "MONITOR")  # either is a valid response


def test_missing_location_returns_none():
    """No lat/lon → can't dispatch, return None."""
    from mission_planner import MissionPlanner
    p = MissionPlanner()
    plan = p.plan({"class": "smoke", "confidence": 0.95})
    assert plan is None


def test_rationale_contains_class():
    from mission_planner import MissionPlanner
    p = MissionPlanner()
    plan = p.plan({"class": "smoke", "confidence": 0.88, "lat": 34.0, "lon": -118.0})
    assert plan is not None
    assert "smoke" in plan.rationale.lower()


def test_priority_mapping():
    from mission_planner import MissionPlanner
    p = MissionPlanner()
    critical = p.plan({"class": "fire", "confidence": 0.90, "lat": 34.0, "lon": -118.0})
    high = p.plan({"class": "fire", "confidence": 0.75, "lat": 34.0, "lon": -118.0})
    assert critical.priority == "CRITICAL"
    assert high.priority == "HIGH"


def test_singleton_get_planner():
    from mission_planner import get_planner, MissionPlanner
    p1 = get_planner()
    p2 = get_planner()
    assert p1 is p2
    assert isinstance(p1, MissionPlanner)
