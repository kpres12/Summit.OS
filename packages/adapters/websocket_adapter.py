"""
Summit.OS — Generic WebSocket Adapter
=======================================

Subscribes to any WebSocket feed and translates messages into Summit.OS
observations.

Modern connected hardware increasingly exposes a WebSocket API: drones,
robots, connected vehicles, smart city infrastructure, IoT gateways,
digital twins, and custom telemetry servers. This adapter handles all of
them with a configurable field-mapping layer.

Supports
--------
- Any JSON-over-WebSocket stream
- Optional auth: Bearer token header or query-parameter token
- Configurable field mapping (same aliases as the Webhook adapter)
- Automatic reconnect with backoff (inherited from framework)
- Optional subscription message sent on connect (e.g. for feed registration)

Dependencies
------------
    pip install websockets

Config extras
-------------
url             : str   — WebSocket URL, e.g. "ws://192.168.1.100:8080/telemetry"
auth_token      : str   — Bearer token (sent as Authorization header)
auth_query_param: str   — Token sent as ?token=... query param instead
subscribe_msg   : str   — JSON string to send immediately after connect (optional)
field_map       : dict  — map source field names → Summit.OS field names
                          e.g. {"vehicle_id": "entity_id", "gps_lat": "lat"}
entity_type     : str   — default entity type if not in message (default ASSET)
entity_id_field : str   — which source field to use as entity_id (default "id")
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("summit.adapters.websocket")

try:
    import websockets
    _WS_AVAILABLE = True
except ImportError:
    websockets = None  # type: ignore
    _WS_AVAILABLE = False

# Common field name aliases → canonical field
_ALIASES = {
    "lat":   ["lat", "latitude", "y", "Lat", "LAT"],
    "lon":   ["lon", "longitude", "lng", "x", "Long", "LON"],
    "alt":   ["alt", "altitude", "elevation", "z", "ALT"],
    "speed": ["speed", "speed_mps", "velocity", "spd"],
    "heading": ["heading", "heading_deg", "bearing", "course", "hdg"],
    "callsign": ["callsign", "name", "label", "display_name", "vehicleName"],
    "entity_type": ["entity_type", "type", "vehicle_type", "assetType"],
}


def _resolve(data: dict, aliases: list) -> Optional[object]:
    for key in aliases:
        if key in data:
            return data[key]
    return None


class WebSocketAdapter(BaseAdapter):
    """
    Subscribes to a WebSocket stream and emits observations from JSON messages.
    """

    adapter_type = "websocket"

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        ex = config.extra
        self._url: str = ex.get("url", "")
        self._auth_token: Optional[str] = ex.get("auth_token")
        self._auth_query: Optional[str] = ex.get("auth_query_param")
        self._subscribe_msg: Optional[str] = ex.get("subscribe_msg")
        self._field_map: dict = ex.get("field_map", {})
        self._default_entity_type: str = ex.get("entity_type", "ASSET")
        self._entity_id_field: str = ex.get("entity_id_field", "id")
        self._ws = None

    async def connect(self) -> None:
        if not _WS_AVAILABLE:
            raise RuntimeError("websockets not installed. Run: pip install websockets")
        if not self._url:
            raise ValueError("WebSocket adapter requires 'url' in config.extra")

        url = self._url
        if self._auth_query:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}token={self._auth_query}"

        headers = {}
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"

        self._ws = await websockets.connect(url, extra_headers=headers)

        if self._subscribe_msg:
            await self._ws.send(self._subscribe_msg)

        logger.info("WebSocket connected: %s", self._url)

    async def disconnect(self) -> None:
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None

    async def stream_observations(self) -> AsyncIterator[dict]:
        async for raw_msg in self._ws:
            if self._stop_event.is_set():
                break
            try:
                data = json.loads(raw_msg)
                if isinstance(data, list):
                    for item in data:
                        obs = self._to_obs(item)
                        if obs:
                            yield obs
                elif isinstance(data, dict):
                    obs = self._to_obs(data)
                    if obs:
                        yield obs
            except json.JSONDecodeError:
                pass  # binary or non-JSON frame

    def _to_obs(self, data: dict) -> Optional[dict]:
        # Apply field map renames first
        mapped = {}
        for src, dst in self._field_map.items():
            if src in data:
                mapped[dst] = data.pop(src)
        data.update(mapped)

        now = datetime.now(timezone.utc)

        # Resolve entity identity
        entity_id = str(
            data.get(self._entity_id_field)
            or data.get("entity_id")
            or data.get("id")
            or self.config.adapter_id
        )

        lat = _resolve(data, _ALIASES["lat"])
        lon = _resolve(data, _ALIASES["lon"])
        alt = _resolve(data, _ALIASES["alt"])
        speed = _resolve(data, _ALIASES["speed"])
        heading = _resolve(data, _ALIASES["heading"])
        callsign = _resolve(data, _ALIASES["callsign"]) or entity_id
        entity_type = (
            data.get("entity_type")
            or _resolve(data, _ALIASES["entity_type"])
            or self._default_entity_type
        )

        # Strip resolved fields from metadata
        resolved_keys = set()
        for aliases in _ALIASES.values():
            resolved_keys.update(aliases)
        resolved_keys.update([self._entity_id_field, "entity_id", "id"])
        metadata = {k: v for k, v in data.items() if k not in resolved_keys}

        obs: dict = {
            "source_id": f"{entity_id}:{now.timestamp():.3f}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": entity_id,
            "callsign": str(callsign),
            "entity_type": str(entity_type),
            "classification": "websocket_asset",
            "ts_iso": now.isoformat(),
            "metadata": metadata,
        }
        if lat is not None and lon is not None:
            obs["position"] = {
                "lat": float(lat),
                "lon": float(lon),
                "alt_m": float(alt) if alt is not None else None,
            }
        if speed is not None or heading is not None:
            obs["velocity"] = {
                "heading_deg": float(heading) if heading is not None else None,
                "speed_mps": float(speed) if speed is not None else None,
                "vertical_mps": None,
            }
        return obs
