"""Mission management endpoints."""
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import text

import state
from models import MissionCreateRequest, MissionAssignment, MissionResponse
from tables import missions, mission_assignments
from helpers import _require_auth, _get_org_id, _publish_mission_update, _safe_json, _safe_isoformat
from planning import _validate_policies, _plan_assignments

router = APIRouter()
logger = logging.getLogger("tasking")


@router.post("/api/v1/missions", response_model=MissionResponse)
async def create_mission(req: MissionCreateRequest, request: Request):
    await _require_auth(request)
    assert state.SessionLocal is not None

    mission_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)

    # ── State machine: PLANNING ──
    sm = None
    if state.STATE_MACHINE_AVAILABLE and state.mission_registry:
        sm = state.mission_registry.create(mission_id)
        # Transitions: PLANNING → POLICY_CHECK

    # Policy validation (org-scoped)
    org_id = _get_org_id(request)

    if sm:
        from apps.tasking.state_machine import MissionState
        sm.transition(MissionState.POLICY_CHECK)  # → POLICY_CHECK

    violations = await _validate_policies(req, org_id)
    policy_ok = len(violations) == 0
    if not policy_ok:
        if sm:
            from apps.tasking.state_machine import MissionState
            sm.transition(MissionState.DENIED)  # → DENIED
        raise HTTPException(status_code=400, detail={"policy_violations": violations})

    # Fetch available assets
    async with state.SessionLocal() as session:
        result = await session.execute(
            text(
                "SELECT asset_id, type, capabilities, battery, link, constraints FROM assets ORDER BY updated_at DESC NULLS LAST"
            )
        )
        assets_rows = [dict(r._mapping) for r in result.all()]

    # Simple availability filter: battery >= 30 and link == 'OK'
    available_assets = [
        a
        for a in assets_rows
        if (a.get("battery") or 0) >= 30
        and (a.get("link") in ("OK", "GOOD", "CONNECTED", None))
    ]
    if not available_assets:
        if sm:
            from apps.tasking.state_machine import MissionState
            sm.transition(MissionState.FAILED)  # → FAILED
        raise HTTPException(
            status_code=409, detail="No available assets to plan mission"
        )

    # ── Role decomposition: assign each hardware domain its own behavior ──
    # Cameras get surveillance, drones get search, subs get underwater grid, etc.
    # Falls back to flat assignment if decomposer unavailable.
    role_brief = []
    try:
        from role_decomposer import RoleDecomposer
        intent = (req.planning_params or {}).get("intent") or (
            req.objectives[0] if req.objectives else "survey"
        )
        decomposer = RoleDecomposer()
        manifest = decomposer.decompose(
            intent=intent,
            available_assets=available_assets,
            area=req.area,
            planning_params=req.planning_params,
        )
        role_brief = manifest.to_console_brief()
        logger.info("Mission %s role manifest: %s", mission_id, manifest.summary())

        # Build assignments_map by running planner per role group
        assignments_map: Dict[str, Any] = {}
        for role in manifest.roles:
            if not role.assets:
                continue
            # Fixed assets (cameras, mesh) get a minimal "activate" plan
            if role.planning_params.get("fixed"):
                for asset in role.assets:
                    assignments_map[asset["asset_id"]] = {
                        "role": role.role_name,
                        "domain": role.domain,
                        "behavior": role.behavior,
                        "sensors_active": role.sensors_active,
                        "pattern": role.pattern,
                        "description": role.description,
                        "waypoints": [],  # fixed — no movement
                    }
                continue

            # Mobile assets — build a per-role MissionCreateRequest copy
            import copy
            role_req = copy.copy(req)
            # Merge role planning params into request.
            # Explicit user params take priority: only fill in role defaults
            # for keys the user didn't set (so pattern="grid" is preserved).
            merged_params = dict(req.planning_params or {})
            role_defaults = {
                "pattern": role.pattern,
                "altitude": role.altitude_m,
                "speed": role.speed_mps,
                "intent": intent,
                "sensors_active": role.sensors_active,
            }
            for k, v in role_defaults.items():
                merged_params.setdefault(k, v)
            role_req.planning_params = merged_params

            role_assignments = await _plan_assignments(role_req, role.assets)
            for asset_id, plan in role_assignments.items():
                plan["role"] = role.role_name
                plan["domain"] = role.domain
                plan["description"] = role.description
                plan["sensors_active"] = role.sensors_active
                assignments_map[asset_id] = plan

    except ImportError:
        # Role decomposer not available — fall back to flat assignment
        if state.ASSIGNMENT_ENGINE_AVAILABLE and req.area and req.area.get("center"):
            from apps.tasking.assignment_engine import AssignmentEngine
            intent = (req.planning_params or {}).get("intent", "survey")
            center = req.area.get("center", {})
            ae = AssignmentEngine()
            assignment_result = ae.assign(
                intent=intent,
                target_lat=float(center.get("lat", 0)),
                target_lon=float(center.get("lon", 0)),
                num_assets=req.num_drones or 1,
                available_assets=available_assets,
                org_id=org_id,
            )
            if assignment_result.selected_assets:
                scored_ids = [s.asset_id for s in assignment_result.selected_assets]
                scored_map = {a["asset_id"]: a for a in available_assets}
                available_assets = [
                    scored_map[aid] for aid in scored_ids if aid in scored_map
                ]
        assignments_map = await _plan_assignments(req, available_assets)
    except Exception as _re:
        logger.warning("Role decomposer error (%s) — falling back to flat plan", _re)
        assignments_map = await _plan_assignments(req, available_assets)

    # ── State machine: DISPATCHED ──
    if sm:
        from apps.tasking.state_machine import MissionState
        sm.transition(MissionState.DISPATCHED)  # → DISPATCHED

    # Persist mission and assignments
    async with state.SessionLocal() as session:
        await session.execute(
            missions.insert().values(
                mission_id=mission_id,
                name=req.name,
                objectives=req.objectives,
                area=req.area,
                policy_ok=policy_ok,
                status="ACTIVE",
                created_at=created_at,
                started_at=created_at,
                org_id=org_id,
            )
        )
        for asset_id, plan in assignments_map.items():
            await session.execute(
                mission_assignments.insert().values(
                    mission_id=mission_id,
                    asset_id=asset_id,
                    plan=plan,
                    status="ASSIGNED",
                    org_id=org_id,
                )
            )
        await session.commit()

    # ── State machine: ACTIVE ──
    if sm:
        from apps.tasking.state_machine import MissionState
        sm.transition(MissionState.ACTIVE)  # → ACTIVE

    # Emit MQTT events and dispatch to each asset
    await _publish_mission_update(
        mission_id,
        {"event": "MISSION_CREATED", "name": req.name, "objectives": req.objectives},
    )

    if state.METRIC_MISSIONS_CREATED:
        state.METRIC_MISSIONS_CREATED.inc()
    if state.METRIC_MISSIONS_ACTIVE:
        state.METRIC_MISSIONS_ACTIVE.inc()

    # Dispatch plans to per-asset task topics (with pre-dispatch OPA check)
    if state.mqtt_client:
        for asset_id, plan in assignments_map.items():
            # Pre-dispatch policy gate
            try:
                from apps.tasking.opa import OPAClient

                opa = OPAClient()
                dispatch_result = await opa.evaluate_pre_dispatch(
                    mission_id=mission_id,
                    asset_id=asset_id,
                    plan=plan,
                    org_id=org_id,
                )
                if not dispatch_result.get("allow", True):
                    logger.warning(
                        f"Pre-dispatch denied for {asset_id}: {dispatch_result.get('deny_reasons')}"
                    )
                    continue
            except Exception:
                pass  # If OPA check fails, proceed (fail-open for dispatch)

            topic = f"tasks/{asset_id}/dispatch"
            message = {
                "task_id": f"mission:{mission_id}",
                "action": "MISSION_EXECUTE",
                "waypoints": plan.get("waypoints", []),
                "plan": plan,
                "ts_iso": datetime.now(timezone.utc).isoformat(),
            }
            state.mqtt_client.publish(topic, json.dumps(message), qos=1)
            # Emit assignment update
            await _publish_mission_update(
                mission_id,
                {"event": "ASSIGNED", "asset_id": asset_id, "plan": plan},
            )

    # Build response
    assignments = [
        MissionAssignment(asset_id=k, plan=v, status="ASSIGNED")
        for k, v in assignments_map.items()
    ]
    return MissionResponse(
        mission_id=mission_id,
        name=req.name,
        objectives=req.objectives,
        status="ACTIVE",
        policy_ok=policy_ok,
        assignments=assignments,
        role_brief=role_brief,
        created_at=created_at,
        started_at=created_at,
    )


@router.get("/api/v1/missions/{mission_id}")
async def get_mission(mission_id: str):
    assert state.SessionLocal is not None
    async with state.SessionLocal() as session:
        res = await session.execute(
            text(
                "SELECT mission_id, name, objectives, area, policy_ok, status, created_at, started_at, completed_at FROM missions WHERE mission_id = :mid"
            ),
            {"mid": mission_id},
        )
        mrow = res.first()
        if not mrow:
            raise HTTPException(status_code=404, detail="Mission not found")
        ares = await session.execute(
            text(
                "SELECT asset_id, plan, status FROM mission_assignments WHERE mission_id = :mid"
            ),
            {"mid": mission_id},
        )
        assignments = [
            MissionAssignment(
                asset_id=r.asset_id, plan=_safe_json(r.plan) or {}, status=r.status
            )
            for r in ares.all()
        ]
        m = dict(mrow._mapping)
        return {
            "mission_id": m["mission_id"],
            "name": m.get("name"),
            "objectives": _safe_json(m.get("objectives")) or [],
            "status": m.get("status"),
            "policy_ok": bool(m.get("policy_ok")),
            "assignments": [a.model_dump() for a in assignments],
            "created_at": _safe_isoformat(
                m.get("created_at") or datetime.now(timezone.utc)
            ),
            "started_at": (
                _safe_isoformat(m.get("started_at")) if m.get("started_at") else None
            ),
            "completed_at": _safe_isoformat(m.get("completed_at")),
        }


@router.get("/api/v1/missions")
async def list_missions(request: Request, limit: int = 50, status: Optional[str] = None):
    assert state.SessionLocal is not None
    org_id = _get_org_id(request)
    conditions = []
    params: Dict[str, Any] = {"lim": limit}
    if status:
        conditions.append("status = :st")
        params["st"] = status
    if state._ENTERPRISE_MULTI_TENANT:
        conditions.append("org_id = :org_id")
        params["org_id"] = org_id
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    query = f"SELECT mission_id, name, objectives, status, created_at, started_at, completed_at FROM missions {where} ORDER BY id DESC LIMIT :lim"
    async with state.SessionLocal() as session:
        res = await session.execute(text(query), params)
        rows = res.all()
        return [
            {
                "mission_id": r.mission_id,
                "name": r.name,
                "objectives": _safe_json(r.objectives) or [],
                "status": r.status,
                "created_at": _safe_isoformat(r.created_at),
                "started_at": _safe_isoformat(r.started_at),
                "completed_at": _safe_isoformat(r.completed_at),
            }
            for r in rows
        ]
