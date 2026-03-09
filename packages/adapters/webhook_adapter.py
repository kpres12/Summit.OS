"""
Summit.OS — Inbound HTTP Webhook Adapter
==========================================

Starts a lightweight aiohttp HTTP server that accepts POSTed JSON observations.
Any system that can POST JSON becomes a Summit.OS signal source.

POST /ingest   — ingest an observation
GET  /health   — returns {"status": "ok"}

Dependencies
------------
    pip install aiohttp
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

try:
    import aiohttp
    from aiohttp import web
except ImportError:
    raise ImportError(
        "aiohttp is required for WebhookAdapter. Install with: pip install aiohttp>=3.9.0"
    )

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("summit.adapters.webhook")

# Field name aliases for common positional data
_LAT_ALIASES = ("lat", "latitude", "y", "Lat", "LAT", "LATITUDE")
_LON_ALIASES = ("lon", "lng", "longitude", "x", "Lon", "LON", "LNG", "LONGITUDE")
_ALT_ALIASES = ("alt_m", "alt", "altitude", "elevation", "elev")


def _extract_float(payload: dict, *keys) -> Optional[float]:
    for k in keys:
        v = payload.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def _extract_str(payload: dict, *keys) -> Optional[str]:
    for k in keys:
        v = payload.get(k)
        if v is not None:
            return str(v)
    return None


class WebhookAdapter(BaseAdapter):
    """
    Listens on an HTTP port and ingests POSTed JSON observations.

    Config extras
    -------------
    listen_port         : int     (default 8090)
    listen_host         : str     (default "0.0.0.0")
    auth_token          : str     (if set, require Authorization: Bearer <token>)
    entity_type_default : str     (default "SENSOR")
    """

    adapter_type = "webhook"

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        ex = config.extra

        self._listen_port: int = int(ex.get("listen_port", 8090))
        self._listen_host: str = ex.get("listen_host", "0.0.0.0")
        self._auth_token: str = ex.get("auth_token", "")
        self._entity_type_default: str = ex.get("entity_type_default", "SENSOR")

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._runner: Optional[web.AppRunner] = None
        self._server_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        app = web.Application()
        app.router.add_post("/ingest", self._handle_ingest)
        app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._listen_host, self._listen_port)
        await site.start()
        self._log.info(
            "Webhook server listening on %s:%d", self._listen_host, self._listen_port
        )

    async def disconnect(self) -> None:
        try:
            if self._runner is not None:
                await self._runner.cleanup()
                self._runner = None
        except Exception:
            pass

    async def stream_observations(self) -> AsyncIterator[dict]:
        while not self._stop_event.is_set():
            try:
                obs = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                yield obs
            except asyncio.TimeoutError:
                continue

    # -------------------------------------------------------------------------
    # HTTP handlers
    # -------------------------------------------------------------------------

    async def _handle_ingest(self, request: web.Request) -> web.Response:
        # Auth check
        if self._auth_token:
            auth_header = request.headers.get("Authorization", "")
            expected = f"Bearer {self._auth_token}"
            if auth_header != expected:
                return web.json_response(
                    {"error": "Unauthorized"}, status=401
                )

        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        obs = self._payload_to_obs(payload)
        if obs is None:
            return web.json_response({"error": "Could not parse observation"}, status=422)

        try:
            self._queue.put_nowait(obs)
        except asyncio.QueueFull:
            self._log.warning("Webhook queue full — dropping observation")
            return web.json_response({"error": "Queue full"}, status=503)

        return web.json_response({"status": "accepted"}, status=202)

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "adapter_id": self.config.adapter_id})

    # -------------------------------------------------------------------------
    # Payload normalisation
    # -------------------------------------------------------------------------

    def _payload_to_obs(self, payload: dict) -> Optional[dict]:
        if not isinstance(payload, dict):
            return None

        entity_id = _extract_str(payload, "entity_id", "id", "device_id", "sensor_id")
        if not entity_id:
            entity_id = f"{self.config.adapter_id}-unknown"

        callsign = _extract_str(payload, "callsign", "name", "label", "device_name")
        entity_type = _extract_str(payload, "entity_type", "type", "device_type")
        if not entity_type:
            entity_type = self._entity_type_default

        lat = _extract_float(payload, *_LAT_ALIASES)
        lon = _extract_float(payload, *_LON_ALIASES)
        alt_m = _extract_float(payload, *_ALT_ALIASES)

        position = None
        if lat is not None and lon is not None:
            position = {"lat": lat, "lon": lon, "alt_m": alt_m}

        heading = _extract_float(payload, "heading", "heading_deg", "course")
        speed = _extract_float(payload, "speed", "speed_mps", "velocity")
        velocity = None
        if heading is not None or speed is not None:
            velocity = {
                "heading_deg": heading,
                "speed_mps": speed,
                "vertical_mps": _extract_float(payload, "vertical_mps", "climb_rate"),
            }

        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        # Carry through any unrecognised keys as metadata
        known_keys = {
            "entity_id", "id", "device_id", "sensor_id",
            "callsign", "name", "label", "device_name",
            "entity_type", "type", "device_type",
            "metadata",
            *_LAT_ALIASES, *_LON_ALIASES, *_ALT_ALIASES,
            "heading", "heading_deg", "course",
            "speed", "speed_mps", "velocity", "vertical_mps", "climb_rate",
        }
        for k, v in payload.items():
            if k not in known_keys:
                metadata[k] = v

        now = datetime.now(timezone.utc)
        return {
            "source_id": f"{entity_id}:{now.timestamp():.3f}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": entity_id,
            "callsign": callsign,
            "position": position,
            "velocity": velocity,
            "entity_type": entity_type,
            "classification": _extract_str(payload, "classification"),
            "metadata": metadata,
            "ts_iso": now.isoformat(),
        }
