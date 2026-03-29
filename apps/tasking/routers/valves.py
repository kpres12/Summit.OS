"""Valve command endpoint."""
import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Request

import state
from models import ValveCommand
from helpers import _require_auth

router = APIRouter()


@router.post("/api/v1/valves/{asset_id}/command")
async def valve_command(asset_id: str, cmd: ValveCommand, request: Request):
    """Publish a valve command to MQTT topic valves/{asset_id}/command (Plainview ValveOps)."""
    await _require_auth(request)
    assert state.mqtt_client is not None
    payload = {
        "command": cmd.command,
        "params": cmd.params or {},
        "safety": cmd.safety or {},
        "mission_id": cmd.mission_id,
        "request_id": cmd.request_id or str(uuid.uuid4()),
        "ts_iso": datetime.now(timezone.utc).isoformat(),
    }
    topic = f"valves/{asset_id}/command"
    state.mqtt_client.publish(topic, json.dumps(payload), qos=1)
    return {"status": "sent", "asset_id": asset_id, "request_id": payload["request_id"]}
