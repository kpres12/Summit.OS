"""Task dispatch and management endpoints."""
import json
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import text

import state
from models import TaskDispatchRequest, Task
from tables import tasks
from helpers import _require_auth

router = APIRouter()


@router.post("/dispatch")
async def dispatch_task(req: TaskDispatchRequest, request: Request):
    await _require_auth(request)
    assert state.SessionLocal is not None
    assert state.mqtt_client is not None

    created_at = datetime.now(timezone.utc)

    # Store task in DB
    async with state.SessionLocal() as session:
        await session.execute(
            tasks.insert().values(
                task_id=req.task_id,
                asset_id=req.asset_id,
                action=req.action,
                status="ACTIVE",
                created_at=created_at,
                started_at=created_at,
            )
        )
        await session.commit()

    # Publish task to MQTT for edge agent to consume
    task_message = {
        "task_id": req.task_id,
        "action": req.action,
        "waypoints": req.waypoints,
        "ts_iso": created_at.isoformat(),
    }

    topic = f"tasks/{req.asset_id}/dispatch"
    state.mqtt_client.publish(topic, json.dumps(task_message), qos=1)

    return {
        "task_id": req.task_id,
        "status": "ACTIVE",
        "message": f"Task dispatched to {req.asset_id}",
    }


@router.get("/tasks/{task_id}", response_model=Task)
async def get_task(task_id: str):
    """Get task status by ID."""
    assert state.SessionLocal is not None

    async with state.SessionLocal() as session:
        result = await session.execute(
            text(
                "SELECT task_id, asset_id, action, status, created_at, started_at, completed_at FROM tasks WHERE task_id = :tid"
            ),
            {"tid": task_id},
        )
        row = result.first()

        if not row:
            raise HTTPException(status_code=404, detail="Task not found")

        return Task(**dict(row._mapping))


@router.get("/tasks", response_model=List[Task])
async def list_tasks(
    asset_id: Optional[str] = None, status: Optional[str] = None, limit: int = 50
):
    """List tasks, optionally filtered by asset_id or status."""
    assert state.SessionLocal is not None

    query = "SELECT task_id, asset_id, action, status, created_at, started_at, completed_at FROM tasks"
    params = {"lim": limit}
    where_clauses = []

    if asset_id:
        where_clauses.append("asset_id = :aid")
        params["aid"] = asset_id

    if status:
        where_clauses.append("status = :st")
        params["st"] = status

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    query += " ORDER BY id DESC LIMIT :lim"

    async with state.SessionLocal() as session:
        result = await session.execute(text(query), params)
        rows = result.all()

        return [Task(**dict(r._mapping)) for r in rows]


@router.post("/tasks/{task_id}/complete")
async def complete_task(task_id: str):
    """Mark a task as completed."""
    assert state.SessionLocal is not None

    async with state.SessionLocal() as session:
        await session.execute(
            tasks.update()
            .where(tasks.c.task_id == task_id)
            .values(status="COMPLETED", completed_at=datetime.now(timezone.utc))
        )
        await session.commit()

    return {"task_id": task_id, "status": "COMPLETED"}


@router.post("/tasks/{task_id}/fail")
async def fail_task(task_id: str, reason: str = "Unknown error"):
    """Mark a task as failed."""
    assert state.SessionLocal is not None

    async with state.SessionLocal() as session:
        await session.execute(
            tasks.update()
            .where(tasks.c.task_id == task_id)
            .values(status="FAILED", completed_at=datetime.now(timezone.utc))
        )
        await session.commit()

    return {"task_id": task_id, "status": "FAILED", "reason": reason}
