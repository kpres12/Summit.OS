"""
apps/tasking/routers/canvas.py — CANVAS BAA feature endpoints.

POST /api/v1/missions/simulate
    Dry-run a mission plan — returns battery/time predictions per asset
    and an overall success probability. No hardware is touched.

POST /api/v1/missions/hierarchical
    Create a mission using multi-tier hierarchical intent decomposition.
    Commander intent flows: COMMANDER → WING_LEAD → FLIGHT_LEAD → ASSET.
    Returns the full tier structure alongside the assignment map.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

import state
from helpers import _require_auth, _get_org_id
from models import MissionCreateRequest

router = APIRouter()
logger = logging.getLogger("tasking.canvas")


# ── Request / response models ─────────────────────────────────────────────────

class SimulateRequest(BaseModel):
    area:            Optional[Dict[str, Any]] = None
    num_drones:      Optional[int]            = None
    objectives:      Optional[List[str]]      = None
    planning_params: Optional[Dict[str, Any]] = None


class SimulateResponse(BaseModel):
    mission_viable:          bool
    success_probability:     float
    estimated_duration_secs: float
    warnings:                List[str]
    recommendations:         List[str]
    coverage_pct:            Optional[float]
    asset_reports:           List[Dict[str, Any]]


class HierarchicalMissionRequest(BaseModel):
    name:            str
    intent:          str
    area:            Optional[Dict[str, Any]] = None
    objectives:      Optional[List[str]]      = None
    planning_params: Optional[Dict[str, Any]] = None
    tiers:           Optional[List[str]]      = None   # e.g. ["WING_LEAD", "FLIGHT_LEAD"]
    dry_run:         bool = False                       # if True, simulate only — don't persist


# ── POST /api/v1/missions/simulate ───────────────────────────────────────────

@router.post("/api/v1/missions/simulate", response_model=SimulateResponse)
async def simulate_mission(req: SimulateRequest, request: Request):
    """
    Dry-run a mission plan against current asset state.

    Simulates battery drain, flight time, and waypoint completion
    for every available asset. Returns predictions + warnings
    without creating a real mission or touching hardware.
    """
    await _require_auth(request)
    assert state.SessionLocal is not None

    from sqlalchemy import text
    from planning import _plan_assignments
    from mission_simulator import MissionSimulator

    # Build a minimal MissionCreateRequest for planning
    plan_req = MissionCreateRequest(
        name             = "sim-preview",
        objectives       = req.objectives or ["simulate"],
        area             = req.area,
        num_drones       = req.num_drones,
        planning_params  = req.planning_params,
    )

    # Fetch available assets
    async with state.SessionLocal() as session:
        result = await session.execute(
            text("SELECT asset_id, type, capabilities, battery, link, constraints FROM assets ORDER BY updated_at DESC NULLS LAST")
        )
        assets = [dict(r._mapping) for r in result.all()]

    available = [
        a for a in assets
        if (a.get("battery") or 0) >= 20
        and a.get("link") in ("OK", "GOOD", "CONNECTED", None)
    ]

    if not available:
        raise HTTPException(status_code=409, detail="No available assets for simulation")

    # Run role decomposer if intent provided
    assignments_map: Dict[str, Any] = {}
    intent = (req.planning_params or {}).get("intent", "survey")
    try:
        from role_decomposer import RoleDecomposer
        decomposer = RoleDecomposer()
        manifest = decomposer.decompose(
            intent           = intent,
            available_assets = available,
            area             = req.area,
            planning_params  = req.planning_params,
        )
        for role in manifest.roles:
            if not role.assets:
                continue
            if role.planning_params.get("fixed"):
                for asset in role.assets:
                    assignments_map[asset["asset_id"]] = {
                        **role.planning_params,
                        "role":     role.role_name,
                        "domain":   role.domain,
                        "priority": role.priority,
                        "waypoints": [],
                    }
                continue
            import copy
            role_req = copy.copy(plan_req)
            merged = dict(req.planning_params or {})
            for k, v in {
                "pattern":        role.pattern,
                "altitude":       role.altitude_m,
                "speed":          role.speed_mps,
                "intent":         intent,
                "sensors_active": role.sensors_active,
            }.items():
                merged.setdefault(k, v)
            role_req.planning_params = merged
            role_assignments = await _plan_assignments(role_req, role.assets)
            for asset_id, plan in role_assignments.items():
                plan["role"]     = role.role_name
                plan["domain"]   = role.domain
                plan["priority"] = role.priority
                assignments_map[asset_id] = plan
    except Exception as exc:
        logger.warning("Role decomposer failed in simulate (%s) — flat plan", exc)
        assignments_map = await _plan_assignments(plan_req, available)

    # Run simulation
    sim = MissionSimulator()
    report = sim.simulate(
        assignments_map = assignments_map,
        assets          = available,
        area            = req.area,
    )

    return SimulateResponse(
        mission_viable          = report.mission_viable,
        success_probability     = report.success_probability,
        estimated_duration_secs = report.estimated_duration_secs,
        warnings                = report.warnings,
        recommendations         = report.recommendations,
        coverage_pct            = report.coverage_pct,
        asset_reports           = [
            {
                "asset_id":        r.asset_id,
                "domain":          r.domain,
                "start_battery":   r.start_battery,
                "battery_after":   r.battery_now,
                "status":          r.status,
                "elapsed_secs":    r.elapsed_secs,
                "distance_m":      r.distance_m,
                "waypoints_done":  r.waypoints_done,
                "waypoints_total": r.waypoints_total,
                "depleted":        r.depleted,
                "depletion_at_wp": r.depletion_at_wp,
            }
            for r in report.asset_reports
        ],
    )


# ── POST /api/v1/missions/hierarchical ───────────────────────────────────────

@router.post("/api/v1/missions/hierarchical")
async def create_hierarchical_mission(req: HierarchicalMissionRequest, request: Request):
    """
    Create a mission using multi-tier hierarchical intent decomposition.

    Intent flows: COMMANDER → WING_LEAD → FLIGHT_LEAD → ASSET
    Each tier filters assets, splits the area, and decomposes intent
    for its sub-group before passing down to the next tier.

    Set dry_run=true to get the tier structure and assignments
    without persisting or dispatching to hardware.
    """
    await _require_auth(request)
    assert state.SessionLocal is not None

    from sqlalchemy import text
    from intent_hierarchy import HierarchicalDecomposer, CommandTier

    # Fetch available assets
    async with state.SessionLocal() as session:
        result = await session.execute(
            text("SELECT asset_id, type, capabilities, battery, link, constraints FROM assets ORDER BY updated_at DESC NULLS LAST")
        )
        assets = [dict(r._mapping) for r in result.all()]

    available = [
        a for a in assets
        if (a.get("battery") or 0) >= 25
        and a.get("link") in ("OK", "GOOD", "CONNECTED", None)
    ]

    if not available:
        raise HTTPException(status_code=409, detail="No available assets")

    # Parse tier chain
    tier_chain = None
    if req.tiers:
        try:
            tier_chain = [CommandTier(t) for t in req.tiers]
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tier: {exc}. Valid values: WING_LEAD, FLIGHT_LEAD"
            )

    # Run hierarchical decomposition
    decomposer = HierarchicalDecomposer()
    try:
        result = decomposer.decompose(
            intent          = req.intent,
            area            = req.area or {},
            assets          = available,
            tiers           = tier_chain,
            planning_params = req.planning_params,
        )
    except Exception as exc:
        logger.error("Hierarchical decomposition failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Decomposition failed: {exc}")

    if req.dry_run:
        return {
            "dry_run":        True,
            "intent":         result.intent,
            "tier_count":     result.tier_count,
            "asset_count":    result.asset_count,
            "tier_structure": result.tier_structure,
            "assignments":    result.assignments,
        }

    # Persist + dispatch — reuse mission creation logic
    org_id     = _get_org_id(request)
    mission_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)

    import json
    from sqlalchemy import text as _text
    from tables import missions, mission_assignments

    async with state.SessionLocal() as session:
        await session.execute(
            missions.insert().values(
                mission_id = mission_id,
                name       = req.name,
                objectives = req.objectives or [req.intent],
                area       = req.area,
                policy_ok  = True,
                status     = "ACTIVE",
                created_at = created_at,
                started_at = created_at,
                org_id     = org_id,
            )
        )
        for asset_id, plan in result.assignments.items():
            await session.execute(
                mission_assignments.insert().values(
                    mission_id = mission_id,
                    asset_id   = asset_id,
                    plan       = plan,
                    status     = "ASSIGNED",
                    org_id     = org_id,
                )
            )
        await session.commit()

    # Dispatch via MQTT
    if state.mqtt_client:
        for asset_id, plan in result.assignments.items():
            topic   = f"tasks/{asset_id}/dispatch"
            message = {
                "task_id":    f"mission:{mission_id}",
                "action":     "MISSION_EXECUTE",
                "waypoints":  plan.get("waypoints", []),
                "plan":       plan,
                "tier":       plan.get("tier"),
                "tier_id":    plan.get("tier_id"),
                "ts_iso":     created_at.isoformat(),
            }
            state.mqtt_client.publish(topic, json.dumps(message), qos=1)

    return {
        "mission_id":     mission_id,
        "name":           req.name,
        "intent":         result.intent,
        "status":         "ACTIVE",
        "tier_count":     result.tier_count,
        "asset_count":    result.asset_count,
        "tier_structure": result.tier_structure,
        "assignments":    [
            {"asset_id": k, "plan": v, "status": "ASSIGNED"}
            for k, v in result.assignments.items()
        ],
        "created_at":     created_at.isoformat(),
    }
