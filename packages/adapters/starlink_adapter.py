"""
Heli.OS — Starlink Adapter
==============================

Integrates Starlink satellite internet terminals into Heli.OS via the
local gRPC API exposed by every Starlink dish on the local network.

This adapter does NOT control Starlink — it reads telemetry from the dish
and surfaces it as a COMMS_NODE entity in Heli.OS. Use cases:

- Track the physical location of mobile command posts and field assets
  equipped with Starlink (the dish reports its own GPS coordinates)
- Monitor field connectivity health — alert operators when a Starlink
  terminal at a fire camp, SAR base, or remote site goes offline or
  degrades below a signal quality threshold
- Map obstruction events (terrain-blocked satellite windows) against
  mission timing to predict comms blackouts
- Asset awareness: know which field teams have active satellite uplinks

Starlink dish local API
-----------------------
Every Starlink terminal exposes a gRPC API on the local network at
192.168.100.1:9200 (no auth required on the local subnet).

Two fallback modes if grpc is unavailable:
1. HTTP status page (some dishes expose http://192.168.100.1/)
2. Poll mode via the official Starlink app's local endpoint

Dependencies
------------
    pip install grpcio grpcio-tools    # for gRPC mode (recommended)
    # OR: no extra deps for HTTP fallback mode

Config extras
-------------
host                : str   — Dish IP (default "192.168.100.1")
port                : int   — gRPC port (default 9200)
terminal_id         : str   — unique identifier for this terminal
terminal_name       : str   — display name (e.g. "BASE-CAMP-ALPHA")
alert_snr_threshold : float — alert if SNR drops below this dB (default 6.0)
alert_on_offline    : bool  — emit ALERT entity type when dish goes offline
poll_interval_seconds: float — telemetry poll rate (default 10.0)
mode                : str   — "grpc" | "http" (default "grpc", falls back to http)
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("heli.adapters.starlink")

# ── gRPC imports (optional) ────────────────────────────────────────────────────
try:
    import grpc
    _GRPC_AVAILABLE = True
except ImportError:
    grpc = None  # type: ignore
    _GRPC_AVAILABLE = False

# ── HTTP fallback ──────────────────────────────────────────────────────────────
try:
    import aiohttp
    _AIOHTTP_AVAILABLE = True
except ImportError:
    _AIOHTTP_AVAILABLE = False

# Starlink gRPC protobuf stubs — generated from SpaceX's published .proto files.
# Install via: pip install starlink-grpc-tools
# https://github.com/sparky8512/starlink-grpc-tools
try:
    import spacex.api.device.device_pb2 as _device_pb2
    import spacex.api.device.device_pb2_grpc as _device_pb2_grpc
    _STARLINK_PROTO_AVAILABLE = True
except ImportError:
    _STARLINK_PROTO_AVAILABLE = False


# ── Minimal protobuf-free gRPC request (raw bytes for GetStatus) ───────────────
# SpaceX publishes the proto schema. This is the serialised GetStatusRequest
# message — works without the generated stubs if grpcio is installed.
_GET_STATUS_METHOD = "/SpaceX.API.Device.Device/Handle"
_STATUS_REQUEST_BYTES = bytes([0x0A, 0x00])  # Handle{get_status: {}}


class StarlinkAdapter(BaseAdapter):
    """
    Reads Starlink dish telemetry and emits COMMS_NODE observations.

    Attempts gRPC first (richest data), falls back to HTTP if unavailable.
    """

    adapter_type = "starlink"

    @classmethod
    def required_extra_fields(cls) -> list[str]:
        return ["terminal_id"]

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        ex = config.extra
        self._host: str = ex.get("host", "192.168.100.1")
        self._port: int = int(ex.get("port", 9200))
        self._terminal_id: str = ex.get("terminal_id", config.adapter_id)
        self._terminal_name: str = ex.get("terminal_name", config.display_name or "Starlink Terminal")
        self._alert_snr: float = float(ex.get("alert_snr_threshold", 6.0))
        self._alert_offline: bool = bool(ex.get("alert_on_offline", True))
        self._mode: str = ex.get("mode", "grpc")
        self._channel = None
        self._stub = None
        self._http_session = None

    # ── Lifecycle ───────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        if self._mode == "grpc" and _GRPC_AVAILABLE:
            self._channel = grpc.aio.insecure_channel(f"{self._host}:{self._port}")
            if _STARLINK_PROTO_AVAILABLE:
                self._stub = _device_pb2_grpc.DeviceStub(self._channel)
            logger.info(
                "Starlink gRPC connected: %s:%d (%s)",
                self._host, self._port,
                "with stubs" if _STARLINK_PROTO_AVAILABLE else "raw channel"
            )
        elif _AIOHTTP_AVAILABLE:
            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            )
            logger.info("Starlink HTTP fallback: http://%s/", self._host)
        else:
            raise RuntimeError(
                "Neither grpcio nor aiohttp is installed.\n"
                "Run: pip install grpcio  OR  pip install aiohttp"
            )

    async def disconnect(self) -> None:
        if self._channel:
            try:
                await self._channel.close()
            except Exception:
                pass
            self._channel = None
        if self._http_session:
            try:
                await self._http_session.close()
            except Exception:
                pass
            self._http_session = None

    async def stream_observations(self) -> AsyncIterator[dict]:
        while not self._stop_event.is_set():
            try:
                if self._channel and _GRPC_AVAILABLE:
                    obs = await self._poll_grpc()
                else:
                    obs = await self._poll_http()
                if obs:
                    yield obs
            except Exception as e:
                logger.warning("Starlink poll error: %s", e)
                raise
            await asyncio.sleep(self.config.poll_interval_seconds)

    # ── gRPC polling ────────────────────────────────────────────────────────────

    async def _poll_grpc(self) -> Optional[dict]:
        if _STARLINK_PROTO_AVAILABLE and self._stub:
            return await self._poll_grpc_stubs()
        return await self._poll_grpc_raw()

    async def _poll_grpc_stubs(self) -> Optional[dict]:
        """Use generated protobuf stubs for richest data."""
        req = _device_pb2.Request(get_status=_device_pb2.GetStatusRequest())
        resp = await self._stub.Handle(req, timeout=5)
        ds = resp.dish_get_status

        lat = ds.gps_stats.latitude if ds.HasField("gps_stats") else None
        lon = ds.gps_stats.longitude if ds.HasField("gps_stats") else None
        snr = ds.snr if ds.HasField("snr") else None

        return self._build_obs(
            lat=lat,
            lon=lon,
            snr=snr,
            uptime_s=ds.device_info.uptime_s,
            downlink_mbps=ds.downlink_throughput_bps / 1e6 if ds.downlink_throughput_bps else None,
            uplink_mbps=ds.uplink_throughput_bps / 1e6 if ds.uplink_throughput_bps else None,
            obstructed=ds.obstruction_stats.currently_obstructed,
            obstruction_pct=ds.obstruction_stats.fraction_obstructed * 100,
            satellites_visible=ds.gps_stats.gps_sats if ds.HasField("gps_stats") else None,
            state="CONNECTED" if snr and snr > self._alert_snr else "DEGRADED",
        )

    async def _poll_grpc_raw(self) -> Optional[dict]:
        """Raw gRPC call without proto stubs — returns minimal obs."""
        try:
            # Issue unary call to the Handle method with raw bytes
            call = self._channel.unary_unary(
                _GET_STATUS_METHOD,
                request_serializer=None,
                response_deserializer=None,
            )
            await call(_STATUS_REQUEST_BYTES, timeout=5)
            # If we get here the dish is reachable — emit a basic heartbeat
            return self._build_obs(state="CONNECTED")
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                return self._build_obs(state="OFFLINE")
            raise

    # ── HTTP fallback ────────────────────────────────────────────────────────────

    async def _poll_http(self) -> Optional[dict]:
        if not self._http_session:
            return self._build_obs(state="UNKNOWN")
        try:
            url = f"http://{self._host}/"
            async with self._http_session.get(url) as resp:
                if resp.status == 200:
                    return self._build_obs(state="CONNECTED")
                return self._build_obs(state="DEGRADED")
        except Exception:
            return self._build_obs(state="OFFLINE")

    # ── Observation builder ─────────────────────────────────────────────────────

    def _build_obs(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        snr: Optional[float] = None,
        uptime_s: Optional[int] = None,
        downlink_mbps: Optional[float] = None,
        uplink_mbps: Optional[float] = None,
        obstructed: bool = False,
        obstruction_pct: Optional[float] = None,
        satellites_visible: Optional[int] = None,
        state: str = "UNKNOWN",
    ) -> dict:
        now = datetime.now(timezone.utc)

        is_alert = (
            (state == "OFFLINE" and self._alert_offline)
            or (snr is not None and snr < self._alert_snr)
            or obstructed
        )
        entity_type = "ALERT" if is_alert else "COMMS_NODE"

        obs: dict = {
            "source_id": f"{self._terminal_id}:{now.timestamp():.3f}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": self._terminal_id,
            "callsign": self._terminal_name,
            "entity_type": entity_type,
            "classification": "starlink_terminal",
            "ts_iso": now.isoformat(),
            "metadata": {
                "connectivity_state": state,
                "host": self._host,
            },
        }

        if lat is not None and lon is not None:
            obs["position"] = {"lat": lat, "lon": lon, "alt_m": None}

        if snr is not None:
            obs["metadata"]["snr_db"] = round(snr, 2)
        if uptime_s is not None:
            obs["metadata"]["uptime_hours"] = round(uptime_s / 3600, 1)
        if downlink_mbps is not None:
            obs["metadata"]["downlink_mbps"] = round(downlink_mbps, 2)
        if uplink_mbps is not None:
            obs["metadata"]["uplink_mbps"] = round(uplink_mbps, 2)
        if obstructed:
            obs["metadata"]["obstructed"] = True
        if obstruction_pct is not None:
            obs["metadata"]["obstruction_pct"] = round(obstruction_pct, 1)
        if satellites_visible is not None:
            obs["metadata"]["gps_satellites"] = satellites_visible

        return obs
