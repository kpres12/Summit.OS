"""WebSocket manager for Summit.OS Data Fabric Service."""

import asyncio
import json
import logging
from typing import List, Dict, Any
from fastapi import WebSocket
from datetime import datetime, timezone

from models import TelemetryMessage, AlertMessage, MissionUpdate


class WebSocketManager:
    """Manages WebSocket connections for real-time data streaming."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_info: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, org_id: str = "default"):
        """Accept new WebSocket connection, tagged with the org_id it belongs to."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_info[websocket] = {
            "connected_at": datetime.now(timezone.utc),
            "subscriptions": set(),
            "client_info": {},
            "org_id": org_id,
        }
        logging.info(
            f"WebSocket connected (org={org_id}). Total connections: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.connection_info.pop(websocket, None)
            logging.info(
                f"WebSocket disconnected. Total connections: {len(self.active_connections)}"
            )

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket connection."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logging.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        """Broadcast message to all connected clients (Community / fallback)."""
        if not self.active_connections:
            return

        tasks = []
        for connection in self.active_connections.copy():
            tasks.append(self._send_to_connection(connection, message))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def broadcast_to_org(self, message: str, org_id: str):
        """
        Broadcast message only to connections belonging to org_id.

        Used in Enterprise multi-tenant mode to prevent cross-org data leakage.
        Falls back to broadcast() when org_id is "default" (Community mode).
        """
        if org_id == "default":
            await self.broadcast(message)
            return

        if not self.active_connections:
            return

        tasks = []
        for connection in self.active_connections.copy():
            conn_org = self.connection_info.get(connection, {}).get("org_id", "default")
            if conn_org == org_id:
                tasks.append(self._send_to_connection(connection, message))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_to_connection(self, websocket: WebSocket, message: str):
        """Send message to a specific connection."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logging.error(f"Error sending to connection: {e}")
            self.disconnect(websocket)

    async def broadcast_telemetry(self, telemetry: TelemetryMessage):
        """Broadcast telemetry data to subscribed clients."""
        message = {
            "type": "telemetry",
            "data": {
                "device_id": telemetry.device_id,
                "timestamp": telemetry.timestamp.isoformat(),
                "location": telemetry.location.model_dump(),
                "sensors": telemetry.sensors,
                "status": telemetry.status,
                "battery_level": telemetry.battery_level,
                "signal_strength": telemetry.signal_strength,
            },
        }

        await self.broadcast(json.dumps(message))

    async def broadcast_alert(self, alert: AlertMessage):
        """Broadcast alert data to subscribed clients."""
        message = {
            "type": "alert",
            "data": {
                "alert_id": alert.alert_id,
                "timestamp": alert.timestamp.isoformat(),
                "severity": alert.severity,
                "location": alert.location.model_dump(),
                "description": alert.description,
                "source": alert.source,
                "category": alert.category,
                "tags": alert.tags,
            },
        }

        await self.broadcast(json.dumps(message))

    async def broadcast_mission_update(self, mission: MissionUpdate):
        """Broadcast mission update to subscribed clients."""
        message = {
            "type": "mission",
            "data": {
                "mission_id": mission.mission_id,
                "timestamp": mission.timestamp.isoformat(),
                "status": mission.status,
                "assets": mission.assets,
                "objectives": mission.objectives,
                "progress": mission.progress,
                "estimated_completion": (
                    mission.estimated_completion.isoformat()
                    if mission.estimated_completion
                    else None
                ),
            },
        }

        await self.broadcast(json.dumps(message))

    async def broadcast_system_status(self, status: Dict[str, Any]):
        """Broadcast system status to subscribed clients."""
        message = {"type": "system_status", "data": status}

        await self.broadcast(json.dumps(message))

    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)

    def get_connection_info(self) -> List[Dict[str, Any]]:
        """Get information about all connections."""
        info = []
        for websocket, conn_info in self.connection_info.items():
            info.append(
                {
                    "connected_at": conn_info["connected_at"].isoformat(),
                    "subscriptions": list(conn_info["subscriptions"]),
                    "client_info": conn_info["client_info"],
                }
            )
        return info
