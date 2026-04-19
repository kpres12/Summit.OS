"""
Heli.OS — AIS Maritime Vessel Tracking Adapter
=================================================

Connects to an AIS feed from either:
  - A physical AIS receiver via NMEA over TCP/serial
  - AISHub REST API (polled)

Emits SURFACE_VESSEL entities for each tracked vessel.

Dependencies
------------
    pip install pyais aiohttp
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Dict, Optional

try:
    import pyais
    from pyais import decode
    from pyais.messages import NMEAMessage
except ImportError:
    raise ImportError(
        "pyais is required for AISAdapter. Install with: pip install pyais"
    )

try:
    import aiohttp
except ImportError:
    raise ImportError(
        "aiohttp is required for AISAdapter. Install with: pip install aiohttp>=3.9.0"
    )

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("heli.adapters.ais")

# AIS navigational status codes
_NAV_STATUS = {
    0: "underway_engine",
    1: "at_anchor",
    2: "not_under_command",
    3: "restricted_maneuverability",
    4: "constrained_by_draught",
    5: "moored",
    6: "aground",
    7: "fishing",
    8: "underway_sailing",
    15: "undefined",
}

# AIS ship type codes (simplified)
_SHIP_TYPES = {
    20: "Wing in ground",
    30: "Fishing",
    31: "Towing",
    32: "Towing large",
    33: "Dredging",
    34: "Diving",
    35: "Military",
    36: "Sailing",
    37: "Pleasure",
    40: "High-speed craft",
    50: "Pilot",
    51: "SAR",
    52: "Tug",
    53: "Port tender",
    55: "Law enforcement",
    60: "Passenger",
    70: "Cargo",
    80: "Tanker",
    90: "Other",
}


def _ship_type_name(code: int) -> str:
    base = (code // 10) * 10
    return _SHIP_TYPES.get(base, _SHIP_TYPES.get(code, f"Unknown({code})"))


class AISAdapter(BaseAdapter):
    """
    Tracks maritime vessels via AIS (Automatic Identification System).

    Config extras
    -------------
    source_type     : "nmea_tcp" | "aishub"   (default "nmea_tcp")
    tcp_host        : str
    tcp_port        : int                      (default 9999)
    aishub_username : str                      (required for aishub)
    bbox            : dict with min_lat/max_lat/min_lon/max_lon
    """

    adapter_type = "ais"

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        ex = config.extra

        self._source_type: str = ex.get("source_type", "nmea_tcp")
        self._tcp_host: str = ex.get("tcp_host", "")
        self._tcp_port: int = int(ex.get("tcp_port", 9999))
        self._aishub_username: str = ex.get("aishub_username", "")
        self._bbox: dict = ex.get(
            "bbox",
            {
                "min_lat": -90,
                "max_lat": 90,
                "min_lon": -180,
                "max_lon": 180,
            },
        )

        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._session: Optional[aiohttp.ClientSession] = None

        # Vessel state cache for merging type 1/2/3 with type 5
        self._vessel_meta: Dict[str, dict] = {}

        # Buffer for multi-sentence AIS messages
        self._sentence_buffer: Dict[int, list] = {}

    async def connect(self) -> None:
        if self._source_type == "nmea_tcp":
            if not self._tcp_host:
                raise ValueError("tcp_host must be set for source_type=nmea_tcp")
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self._tcp_host, self._tcp_port),
                timeout=10.0,
            )
            self._log.info(
                "Connected to AIS NMEA feed at %s:%d", self._tcp_host, self._tcp_port
            )
        elif self._source_type == "aishub":
            if not self._aishub_username:
                raise ValueError("aishub_username is required for source_type=aishub")
            self._session = aiohttp.ClientSession()
            self._log.info("AISHub REST polling mode initialised")
        else:
            raise ValueError(f"Unknown source_type: {self._source_type!r}")

    async def disconnect(self) -> None:
        try:
            if self._writer is not None:
                self._writer.close()
                await self._writer.wait_closed()
                self._writer = None
                self._reader = None
        except Exception:
            pass
        try:
            if self._session is not None:
                await self._session.close()
                self._session = None
        except Exception:
            pass

    async def stream_observations(self) -> AsyncIterator[dict]:
        if self._source_type == "nmea_tcp":
            async for obs in self._stream_nmea():
                yield obs
        else:
            async for obs in self._stream_aishub():
                yield obs

    # -------------------------------------------------------------------------
    # NMEA TCP streaming
    # -------------------------------------------------------------------------

    async def _stream_nmea(self) -> AsyncIterator[dict]:
        while not self._stop_event.is_set():
            try:
                raw = await asyncio.wait_for(self._reader.readline(), timeout=30.0)
            except asyncio.TimeoutError:
                continue
            line = raw.decode("ascii", errors="ignore").strip()
            if not line:
                continue
            obs = await self._parse_nmea_sentence(line)
            if obs:
                yield obs

    async def _parse_nmea_sentence(self, sentence: str) -> Optional[dict]:
        """Parse a single AIS NMEA sentence; return observation or None."""
        if not (sentence.startswith("!AIVDM") or sentence.startswith("!AIVDO")):
            return None
        try:
            msg = NMEAMessage(sentence.encode())
            # Handle multi-part messages
            fill_bits = msg.fill_bits
            count = msg.count
            index = msg.index
            seq = msg.seq_id

            if count == 1:
                decoded = msg.decode()
            else:
                # Buffer multi-sentence messages by sequence id
                key = seq if seq else 0
                if key not in self._sentence_buffer:
                    self._sentence_buffer[key] = []
                self._sentence_buffer[key].append(msg)
                if len(self._sentence_buffer[key]) < count:
                    return None
                parts = self._sentence_buffer.pop(key)
                parts.sort(key=lambda m: m.index)
                decoded = NMEAMessage.assemble_from_iterable(parts).decode()

            return self._decoded_to_obs(decoded)
        except Exception as exc:
            self._log.debug("AIS parse error: %s", exc)
            return None

    def _decoded_to_obs(self, decoded) -> Optional[dict]:
        """Convert a decoded pyais message to an observation dict."""
        mmsi = str(getattr(decoded, "mmsi", "unknown"))
        msg_type = getattr(decoded, "msg_type", 0)

        if msg_type in (1, 2, 3):
            lat = getattr(decoded, "lat", None)
            lon = getattr(decoded, "lon", None)
            if lat is None or lon is None or abs(lat) > 90 or abs(lon) > 180:
                return None
            if not self._in_bbox(lat, lon):
                return None

            sog = getattr(decoded, "speed", None)
            speed_mps = float(sog) * 0.514444 if sog is not None else None
            cog = getattr(decoded, "course", None)
            status_code = getattr(decoded, "status", 15)
            nav_status = _NAV_STATUS.get(int(status_code), "undefined")

            meta = self._vessel_meta.get(mmsi, {})
            meta.update(
                {
                    "mmsi": mmsi,
                    "nav_status": nav_status,
                    "rot": getattr(decoded, "rot", None),
                }
            )
            self._vessel_meta[mmsi] = meta

            callsign = meta.get("vessel_name") or mmsi
            return self._build_obs(mmsi, callsign, lat, lon, speed_mps, cog, meta)

        elif msg_type == 5:
            # Static and voyage data
            vessel_name = (getattr(decoded, "shipname", "") or "").strip()
            ship_type_code = getattr(decoded, "ship_type", 0)
            draught = getattr(decoded, "draught", None)
            destination = (getattr(decoded, "destination", "") or "").strip()

            meta = self._vessel_meta.get(mmsi, {})
            meta.update(
                {
                    "mmsi": mmsi,
                    "vessel_name": vessel_name or mmsi,
                    "ship_type": _ship_type_name(ship_type_code),
                    "ship_type_code": ship_type_code,
                    "draught": float(draught) / 10 if draught else None,
                    "destination": destination,
                }
            )
            self._vessel_meta[mmsi] = meta
            return None  # No position in type 5, don't emit yet

        return None

    # -------------------------------------------------------------------------
    # AISHub REST polling
    # -------------------------------------------------------------------------

    async def _stream_aishub(self) -> AsyncIterator[dict]:
        while not self._stop_event.is_set():
            try:
                vessels = await self._fetch_aishub()
                for obs in vessels:
                    yield obs
            except Exception as exc:
                self._log.warning("AISHub fetch error: %s", exc)
            await self._interruptible_sleep(self.config.poll_interval_seconds)

    async def _fetch_aishub(self) -> list[dict]:
        bbox = self._bbox
        url = (
            f"https://data.aishub.net/ws.php"
            f"?username={self._aishub_username}"
            f"&format=1&output=json&compress=0"
            f"&latmin={bbox['min_lat']}&latmax={bbox['max_lat']}"
            f"&lonmin={bbox['min_lon']}&lonmax={bbox['max_lon']}"
        )
        async with self._session.get(
            url, timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)

        if not isinstance(data, list) or len(data) < 2:
            return []

        results = []
        vessels = data[1] if isinstance(data[1], list) else []
        for v in vessels:
            try:
                lat = float(v.get("LATITUDE", 0))
                lon = float(v.get("LONGITUDE", 0))
                if abs(lat) > 90 or abs(lon) > 180:
                    continue
                mmsi = str(v.get("MMSI", "unknown"))
                sog = v.get("SOG")
                speed_mps = float(sog) * 0.514444 if sog is not None else None
                cog = v.get("COG")
                vessel_name = (v.get("NAME", "") or "").strip() or mmsi
                ship_type_code = v.get("SHIPTYPE", 0)
                nav_status_code = v.get("NAVSTAT", 15)

                meta = {
                    "mmsi": mmsi,
                    "vessel_name": vessel_name,
                    "ship_type": _ship_type_name(ship_type_code),
                    "ship_type_code": ship_type_code,
                    "nav_status": _NAV_STATUS.get(nav_status_code, "undefined"),
                    "draught": v.get("DRAUGHT"),
                    "destination": (v.get("DEST", "") or "").strip(),
                    "imo": v.get("IMO"),
                }
                obs = self._build_obs(mmsi, vessel_name, lat, lon, speed_mps, cog, meta)
                results.append(obs)
            except Exception as exc:
                self._log.debug("AISHub vessel parse error: %s", exc)
        return results

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _in_bbox(self, lat: float, lon: float) -> bool:
        b = self._bbox
        return (
            b["min_lat"] <= lat <= b["max_lat"] and b["min_lon"] <= lon <= b["max_lon"]
        )

    def _build_obs(
        self,
        mmsi: str,
        callsign: str,
        lat: float,
        lon: float,
        speed_mps: Optional[float],
        cog: Optional[float],
        meta: dict,
    ) -> dict:
        now = datetime.now(timezone.utc)
        entity_id = f"ais-{mmsi}"
        return {
            "source_id": f"{entity_id}:{now.timestamp():.3f}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": entity_id,
            "callsign": callsign,
            "position": {"lat": lat, "lon": lon, "alt_m": 0.0},
            "velocity": {
                "heading_deg": float(cog) if cog is not None else None,
                "speed_mps": speed_mps,
                "vertical_mps": None,
            },
            "entity_type": "SURFACE_VESSEL",
            "classification": None,
            "metadata": meta,
            "ts_iso": now.isoformat(),
        }
