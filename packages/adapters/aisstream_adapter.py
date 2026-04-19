"""
Heli.OS — AISStream.io Maritime Vessel Adapter
=================================================
Connects to aisstream.io WebSocket for real-time AIS vessel tracking.
Free API key at https://aisstream.io — provides global vessel positions.

What you get:
  - MMSI + vessel name + callsign
  - Lat/lon + COG (course over ground) + SOG (speed over ground)
  - Vessel type (cargo, tanker, passenger, fishing, etc.)
  - Draught, destination, ETA
  - Navigation status (underway, moored, anchored, etc.)

Config (AdapterConfig.extra fields):
  api_key          — aisstream.io API key (required, or AISSTREAM_API_KEY env var)
  bounding_boxes   — list of [[min_lat, min_lon], [max_lat, max_lon]] boxes
                     default: entire world (may be very high volume)

Environment:
  AISSTREAM_API_KEY — API key override

Register in adapters.json:
  {
    "adapter_type": "aisstream",
    "name": "AISStream Maritime",
    "extra": {
      "api_key": "your-key-here",
      "bounding_boxes": [[[32.0, -120.0], [34.5, -117.0]]]
    }
  }
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import AsyncIterator, Optional

try:
    import websockets
except ImportError:
    raise ImportError("websockets is required: pip install websockets")

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("heli.adapters.aisstream")

AISSTREAM_URL = "wss://stream.aisstream.io/v0/stream"

_NAV_STATUS = {
    0: "underway_engine", 1: "anchored", 2: "not_under_command",
    3: "restricted_maneuverability", 4: "constrained_draught",
    5: "moored", 6: "aground", 7: "fishing", 8: "sailing",
}

_VESSEL_TYPE_MAP = {
    # IMO vessel type codes → Heli.OS label
    range(60, 70): "PASSENGER",
    range(70, 80): "CARGO",
    range(80, 90): "TANKER",
    range(30, 32): "FISHING",
    range(36, 38): "SAILING",
    range(50, 60): "PILOT_TUG",
}


def _vessel_label(type_code: int) -> str:
    for r, label in _VESSEL_TYPE_MAP.items():
        if type_code in r:
            return label
    return "VESSEL"


class AISStreamAdapter(BaseAdapter):
    adapter_type = "aisstream"

    def __init__(self, config: AdapterConfig, mqtt_client=None):
        super().__init__(config, mqtt_client)
        ex = config.extra or {}
        self._api_key = ex.get("api_key") or os.getenv("AISSTREAM_API_KEY", "")
        self._bboxes  = ex.get("bounding_boxes", [[[-90, -180], [90, 180]]])
        self._ws      = None

    async def connect(self) -> None:
        if not self._api_key:
            raise ValueError("AISSTREAM_API_KEY not set — get a free key at aisstream.io")
        self._ws = await websockets.connect(AISSTREAM_URL)
        subscribe_msg = {
            "APIkey": self._api_key,
            "BoundingBoxes": self._bboxes,
            "FilterMessageTypes": ["PositionReport", "ShipStaticData"],
        }
        await self._ws.send(json.dumps(subscribe_msg))
        logger.info("AISStream connected, subscribed to %d bbox(es)", len(self._bboxes))

    async def disconnect(self) -> None:
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def stream_observations(self) -> AsyncIterator[dict]:
        assert self._ws is not None
        async for raw in self._ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("MessageType")
            meta     = msg.get("MetaData", {})
            mmsi     = str(meta.get("MMSI", ""))
            if not mmsi:
                continue

            lat  = meta.get("latitude")
            lon  = meta.get("longitude")
            name = meta.get("ShipName", "").strip() or mmsi

            if msg_type == "PositionReport":
                pr = msg.get("Message", {}).get("PositionReport", {})
                cog     = pr.get("Cog", 0)
                sog     = pr.get("Sog", 0)
                nav_s   = _NAV_STATUS.get(pr.get("NavigationalStatus", 0), "unknown")
                heading = pr.get("TrueHeading", cog)

                if lat is None or lon is None:
                    continue

                yield {
                    "entity_id":  f"ais-{mmsi}",
                    "type":       "neutral",
                    "callsign":   name,
                    "position":   {"lat": lat, "lon": lon, "alt": 0},
                    "last_seen":  int(time.time()),
                    "properties": {
                        "asset_type":   "VESSEL",
                        "mmsi":         mmsi,
                        "cog_deg":      round(cog, 1),
                        "sog_kts":      round(sog, 1),
                        "heading":      round(heading, 1),
                        "nav_status":   nav_s,
                        "source":       "aisstream",
                        "controllable": False,
                    },
                }

            elif msg_type == "ShipStaticData":
                sd = msg.get("Message", {}).get("ShipStaticData", {})
                vessel_type = sd.get("Type", 0)
                callsign    = (sd.get("CallSign") or "").strip()
                dest        = (sd.get("Destination") or "").strip()

                yield {
                    "entity_id":  f"ais-{mmsi}",
                    "type":       "neutral",
                    "callsign":   name,
                    "position":   {"lat": lat or 0, "lon": lon or 0, "alt": 0},
                    "last_seen":  int(time.time()),
                    "properties": {
                        "asset_type":   _vessel_label(vessel_type),
                        "mmsi":         mmsi,
                        "callsign":     callsign,
                        "vessel_type":  vessel_type,
                        "destination":  dest,
                        "draught_m":    sd.get("MaximumStaticDraught"),
                        "source":       "aisstream",
                        "controllable": False,
                    },
                }
