"""Tiered response mission endpoints."""
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import text
from packages.schemas.drones import TieredMissionRequest, TieredMissionStatus, MissionTier

import state
from tables import tiered_missions, mission_assignments
from helpers import _require_auth, _get_org_id
from planning import _assess_threat, _select_tiered_assets, _plan_tiered_mission, _create_containment_pattern

router = APIRouter()


@router.post("/api/v1/tiered-missions", response_model=TieredMissionStatus)
async def create_tiered_mission(req: TieredMissionRequest, request: Request):
    """Create a new tiered response mission."""
    await _require_auth(request)
    assert state.SessionLocal is not None

    org_id = _get_org_id(request)
    mission_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)

    # Fetch available assets
    async with state.SessionLocal() as session:
        result = await session.execute(
            text(
                "SELECT asset_id, type, capabilities, battery, link, constraints FROM assets ORDER BY updated_at DESC NULLS LAST"
            )
        )
        assets_rows = [dict(r._mapping) for r in result.all()]

    # Filter for available tiered response assets
    available_assets = [
        a
        for a in assets_rows
        if (a.get("battery") or 0)
        >= 50  # Higher battery requirement for tiered response
        and (a.get("link") in ("OK", "GOOD", "CONNECTED", None))
        and a.get("capabilities", {}).get("drone_type") in ["scout", "interceptor"]
    ]

    if not available_assets:
        raise HTTPException(
            status_code=409, detail="No available tiered response assets"
        )

    # Plan Tier 1 verification mission
    tier_1_plans = await _plan_tiered_mission(req, available_assets)
    tier_1_assets = list(tier_1_plans.keys())

    # Store tiered mission
    async with state.SessionLocal() as session:
        await session.execute(
            tiered_missions.insert().values(
                mission_id=mission_id,
                alert_id=req.alert_id,
                current_tier="tier_1_verify",
                tier_1_status="ACTIVE",
                assets_deployed=tier_1_assets,
                fire_threshold=(
                    req.intervention_threshold.model_dump()
                    if req.intervention_threshold
                    else None
                ),
                created_at=created_at,
                updated_at=created_at,
                org_id=org_id,
            )
        )

        # Create mission assignments
        for asset_id, plan in tier_1_plans.items():
            await session.execute(
                mission_assignments.insert().values(
                    mission_id=mission_id,
                    asset_id=asset_id,
                    plan=plan,
                    status="ASSIGNED",
                )
            )

        await session.commit()

    # Dispatch to assets
    if state.mqtt_client:
        for asset_id, plan in tier_1_plans.items():
            topic = f"tasks/{asset_id}/dispatch"
            message = {
                "task_id": f"tiered:{mission_id}",
                "action": "TIER_1_VERIFY",
                "tier": "tier_1_verify",
                "waypoints": plan.get("waypoints", []),
                "plan": plan,
                "alert_id": req.alert_id,
                "ts_iso": created_at.isoformat(),
            }
            state.mqtt_client.publish(topic, json.dumps(message), qos=1)

    return TieredMissionStatus(
        mission_id=mission_id,
        current_tier=MissionTier.TIER_1_VERIFY,
        tier_1_status="ACTIVE",
        assets_deployed=tier_1_assets,
        created_at=created_at,
        updated_at=created_at,
    )


@router.post("/api/v1/tiered-missions/{mission_id}/escalate")
async def escalate_tiered_mission(
    mission_id: str, verification_data: Dict[str, Any], request: Request
):
    """Escalate tiered mission to next tier based on verification results."""
    await _require_auth(request)
    assert state.SessionLocal is not None

    # Get current mission state
    async with state.SessionLocal() as session:
        res = await session.execute(
            text("SELECT * FROM tiered_missions WHERE mission_id = :mid"),
            {"mid": mission_id},
        )
        mission_row = res.first()
        if not mission_row:
            raise HTTPException(status_code=404, detail="Tiered mission not found")

        mission_data = dict(mission_row._mapping)
        current_tier = mission_data["current_tier"]

        # Assess threat from verification data using generic framework
        target_location = {
            "lat": verification_data.get("lat", 0),
            "lon": verification_data.get("lon", 0),
        }
        domain = verification_data.get("domain", "generic")
        threat_assessment = await _assess_threat(
            target_location, verification_data, domain
        )

        escalation_needed = threat_assessment.escalation_required
        next_tier = None

        if current_tier == "tier_1_verify" and escalation_needed:
            next_tier = "tier_2_suppress"
        elif current_tier == "tier_2_suppress" and threat_assessment.threat_level in [
            "high",
            "critical",
        ]:
            next_tier = "tier_3_contain"

        if not next_tier:
            # No escalation needed, mark current tier as completed
            await session.execute(
                tiered_missions.update()
                .where(tiered_missions.c.mission_id == mission_id)
                .values(
                    tier_1_status=(
                        "COMPLETED"
                        if current_tier == "tier_1_verify"
                        else mission_data["tier_1_status"]
                    ),
                    tier_2_status=(
                        "COMPLETED"
                        if current_tier == "tier_2_suppress"
                        else mission_data["tier_2_status"]
                    ),
                    verification_result=verification_data,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()
            return {
                "escalated": False,
                "reason": "No escalation required",
                "threat_assessment": threat_assessment.model_dump(),
            }

        # Get available assets for next tier
        result = await session.execute(
            text(
                "SELECT asset_id, type, capabilities, battery, link, constraints FROM assets ORDER BY updated_at DESC NULLS LAST"
            )
        )
        assets_rows = [dict(r._mapping) for r in result.all()]

        available_assets = [
            a
            for a in assets_rows
            if (a.get("battery") or 0) >= 40
            and (a.get("link") in ("OK", "GOOD", "CONNECTED", None))
            and a.get("capabilities", {}).get("drone_type") in ["scout", "interceptor"]
        ]

        # Plan next tier
        if next_tier == "tier_2_suppress":
            tier_2_assets = await _select_tiered_assets(
                target_location, MissionTier.TIER_2_SUPPRESS, available_assets
            )

            if not tier_2_assets:
                raise HTTPException(
                    status_code=409, detail="No available assets for Tier 2 suppression"
                )

            # Create intervention plan
            intervention_plans = {}
            for asset in tier_2_assets:
                asset_id = asset["asset_id"]
                capabilities = asset.get("capabilities", {})

                intervention_plans[asset_id] = {
                    "tier": "tier_2_suppress",
                    "role": "intervention",
                    "target_location": target_location,
                    "payload": {
                        "type": "liquid_capsule",  # Generic payload type
                        "capacity": capabilities.get("payload_capacity", 10),
                        "deployment_pattern": "targeted_drop",
                    },
                    "approach_altitude": 40,
                    "drop_altitude": 20,
                    "waypoints": [
                        {
                            "lat": target_location["lat"],
                            "lon": target_location["lon"],
                            "alt": 20,
                            "speed": capabilities.get("max_speed", 60),
                            "action": "DEPLOY_PAYLOAD",
                        }
                    ],
                }

            deployed_assets = mission_data["assets_deployed"] + list(
                intervention_plans.keys()
            )

        elif next_tier == "tier_3_contain":
            containment_assets = await _select_tiered_assets(
                target_location, MissionTier.TIER_3_CONTAIN, available_assets
            )
            intervention_plans = await _create_containment_pattern(
                target_location, containment_assets
            )
            deployed_assets = mission_data["assets_deployed"] + list(
                intervention_plans.keys()
            )

        # Update mission with next tier
        update_values = {
            "current_tier": next_tier,
            "verification_result": verification_data,
            "assets_deployed": deployed_assets,
            "updated_at": datetime.now(timezone.utc),
        }

        if next_tier == "tier_2_suppress":
            update_values["tier_2_status"] = "ACTIVE"
            update_values["tier_1_status"] = "COMPLETED"
        elif next_tier == "tier_3_contain":
            update_values["tier_3_status"] = "ACTIVE"
            update_values["tier_2_status"] = "COMPLETED"

        await session.execute(
            tiered_missions.update()
            .where(tiered_missions.c.mission_id == mission_id)
            .values(**update_values)
        )

        # Create new mission assignments
        for asset_id, plan in intervention_plans.items():
            await session.execute(
                mission_assignments.insert().values(
                    mission_id=mission_id,
                    asset_id=asset_id,
                    plan=plan,
                    status="ASSIGNED",
                )
            )

        await session.commit()

        # Dispatch to new assets
        if state.mqtt_client:
            for asset_id, plan in intervention_plans.items():
                topic = f"tasks/{asset_id}/dispatch"
                message = {
                    "task_id": f"tiered:{mission_id}:{next_tier}",
                    "action": (
                        "TIER_2_SUPPRESS"
                        if next_tier == "tier_2_suppress"
                        else "TIER_3_CONTAIN"
                    ),
                    "tier": next_tier,
                    "waypoints": plan.get("waypoints", []),
                    "plan": plan,
                    "verification_data": verification_data,
                    "threat_assessment": threat_assessment,
                    "ts_iso": datetime.now(timezone.utc).isoformat(),
                }
                state.mqtt_client.publish(topic, json.dumps(message), qos=1)

        return {
            "escalated": True,
            "next_tier": next_tier,
            "assets_deployed": list(intervention_plans.keys()),
            "threat_assessment": threat_assessment.model_dump(),
        }


@router.get("/api/v1/tiered-missions/{mission_id}")
async def get_tiered_mission(mission_id: str):
    """Get tiered mission status."""
    assert state.SessionLocal is not None

    async with state.SessionLocal() as session:
        res = await session.execute(
            text("SELECT * FROM tiered_missions WHERE mission_id = :mid"),
            {"mid": mission_id},
        )
        mission_row = res.first()
        if not mission_row:
            raise HTTPException(status_code=404, detail="Tiered mission not found")

        mission_data = dict(mission_row._mapping)

        # Get assignments
        ares = await session.execute(
            text(
                "SELECT asset_id, plan, status FROM mission_assignments WHERE mission_id = :mid"
            ),
            {"mid": mission_id},
        )
        assignments = [dict(r._mapping) for r in ares.all()]

        return {
            "mission_id": mission_data["mission_id"],
            "alert_id": mission_data["alert_id"],
            "current_tier": mission_data["current_tier"],
            "tier_1_status": mission_data["tier_1_status"],
            "tier_2_status": mission_data["tier_2_status"],
            "tier_3_status": mission_data["tier_3_status"],
            "verification_result": mission_data["verification_result"],
            "intervention_result": mission_data["intervention_result"],
            "escalation_reason": mission_data["escalation_reason"],
            "assets_deployed": mission_data["assets_deployed"],
            "assignments": assignments,
            "created_at": (
                mission_data["created_at"].isoformat()
                if mission_data["created_at"]
                else None
            ),
            "updated_at": (
                mission_data["updated_at"].isoformat()
                if mission_data["updated_at"]
                else None
            ),
        }


@router.get("/api/v1/tiered-missions")
async def list_tiered_missions(
    limit: int = 50, tier: Optional[str] = None, request: Request = None
):
    """List tiered missions."""
    assert state.SessionLocal is not None

    org_id = _get_org_id(request) if request else "default"

    query = "SELECT mission_id, alert_id, current_tier, tier_1_status, tier_2_status, tier_3_status, created_at, updated_at FROM tiered_missions"
    params: Dict[str, Any] = {"lim": limit}
    conditions = []

    if state._ENTERPRISE_MULTI_TENANT and org_id != "default":
        conditions.append("org_id = :org_id")
        params["org_id"] = org_id

    if tier:
        conditions.append("current_tier = :tier")
        params["tier"] = tier

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY id DESC LIMIT :lim"

    async with state.SessionLocal() as session:
        res = await session.execute(text(query), params)
        rows = res.all()

        return [
            {
                "mission_id": r.mission_id,
                "alert_id": r.alert_id,
                "current_tier": r.current_tier,
                "tier_1_status": r.tier_1_status,
                "tier_2_status": r.tier_2_status,
                "tier_3_status": r.tier_3_status,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "updated_at": r.updated_at.isoformat() if r.updated_at else None,
            }
            for r in rows
        ]
