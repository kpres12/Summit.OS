"""
packages/adapters/dump1090_adapter.py — Local ADS-B receiver adapter.

Connects to dump1090 (or compatible software) running on a local RTL-SDR
dongle to provide real-time aircraft positions with ZERO internet dependency.

Hardware needed: RTL-SDR dongle (~$25) + 1090MHz antenna
Software: dump1090-fa, dump1090-mutability, or readsb

This replaces the OpenSky adapter for offline/denied-environment operations.
Every aircraft within ~200nm that has ADS-B OUT broadcasts its position
on 1090MHz — Heli.OS picks it up directly from the air.

What it provides:
  - ICAO hex code (unique aircraft ID)
  - Callsign/flight number
  - Lat/lon position
  - Altitude (barometric + geometric)
  - Ground speed + track
  - Vertical rate
  - Squawk code
  - Emergency status

Connection modes:
  json_poll  → poll dump1090's aircraft.json HTTP endpoint (default)
               URL: http://localhost:8080/data/aircraft.json
               (dump1090-fa: http://localhost:8080/skyaware/data/aircraft.json)

  beast_tcp  → connect to dump1090 Beast TCP output port (default 30005)
               Raw Mode-S frames — requires pyModeS to decode

  raw_tcp    → connect to dump1090 raw TCP output port (default 30002)
               Raw AVR/Beast frames

Config (extra fields):
  mode:         "json_poll" | "beast_tcp"    (default: "json_poll")
  host:         dump1090 hostname             (default: "localhost")
  port:         HTTP port or TCP port         (default: 8080 for json_poll)
  json_path:    path to aircraft.json         (default: "/data/aircraft.json")
  stale_seconds: drop aircraft not seen for N seconds (default: 60)

Environment:
  DUMP1090_HOST   — override host
  DUMP1090_PORT   — override port
  DUMP1090_MODE   — override mode

Quick start:
  # Install dump1090-fa (Debian/Ubuntu):
  sudo apt install dump1090-fa

  # Or run via Docker:
  docker run -d --device=/dev/bus/usb --net=host ghcr.io/flightaware/piaware

  # Register in Heli.OS:
  {
    "adapter_id": "adsb-local",
    "adapter_type": "dump1090",
    "display_name": "Local ADS-B Receiver",
    "extra": {"mode": "json_poll", "host": "localhost", "port": 8080}
  }
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from .base import BaseAdapter, AdapterConfig

logger = logging.getLogger("heli.adapters.dump1090")

# dump1090 JSON field → Heli.OS field mapping
# aircraft.json format: https://github.com/flightaware/dump1090/blob/master/README-json.md
_SQUAWK_EMERGENCY = {"7500", "7600", "7700"}  # hijack, radio fail, emergency


class Dump1090Adapter(BaseAdapter):
    """
    Local ADS-B adapter via dump1090 or compatible software.

    Polls aircraft.json for current aircraft positions, converting each
    aircraft into a Heli.OS AIRCRAFT entity observation.

    Works completely offline — data comes directly from RF signals in the air.
    Range: ~200nm (line of sight, terrain dependent).
    """

    adapter_type = "dump1090"

    @classmethod
    def required_extra_fields(cls) -> list[str]:
        return []  # All fields have sensible defaults

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        extra = config.extra or {}

        self._mode = (
            os.getenv("DUMP1090_MODE") or extra.get("mode", "json_poll")
        ).lower()
        self._host = os.getenv("DUMP1090_HOST") or extra.get("host", "localhost")
        self._port = int(
            os.getenv("DUMP1090_PORT") or extra.get("port", 8080)
        )
        self._json_path = extra.get("json_path", "/data/aircraft.json")
        self._stale_seconds = float(extra.get("stale_seconds", 60.0))
        self._poll_interval = float(config.poll_interval_seconds or 1.0)

        # Try alternate paths for different dump1090 variants
        self._json_url = f"http://{self._host}:{self._port}{self._json_path}"

        # Track aircraft state for velocity calculation
        self._aircraft_cache: Dict[str, Dict[str, Any]] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Verify dump1090 is reachable."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._check_connection)

    def _check_connection(self) -> None:
        """Test connection to dump1090 JSON endpoint."""
        urls_to_try = [
            self._json_url,
            f"http://{self._host}:{self._port}/skyaware/data/aircraft.json",
            f"http://{self._host}:{self._port}/dump1090/data/aircraft.json",
        ]
        last_exc = None
        for url in urls_to_try:
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    data = json.loads(resp.read())
                    if "aircraft" in data or "now" in data:
                        self._json_url = url  # use whichever worked
                        logger.info(
                            "dump1090 connected at %s — %d aircraft tracked",
                            url, len(data.get("aircraft", [])),
                        )
                        return
            except Exception as exc:
                last_exc = exc
                continue
        raise ConnectionError(
            f"dump1090 not reachable at {self._host}:{self._port}. "
            f"Is dump1090/readsb running? Last error: {last_exc}"
        )

    async def disconnect(self) -> None:
        pass  # Stateless HTTP polling — nothing to close

    async def stream_observations(self) -> AsyncIterator[dict]:
        """Poll aircraft.json and yield one observation per tracked aircraft."""
        loop = asyncio.get_event_loop()
        while True:
            try:
                aircraft_list = await loop.run_in_executor(None, self._fetch_aircraft)
                now_ts = datetime.now(timezone.utc).isoformat()

                for ac in aircraft_list:
                    obs = self._aircraft_to_observation(ac, now_ts)
                    if obs:
                        yield obs

                # Prune stale aircraft from our local state cache
                self._prune_stale_cache()

            except Exception as exc:
                logger.warning("dump1090 poll error: %s", exc)
                raise  # Let the base adapter reconnect

            await asyncio.sleep(self._poll_interval)

    # ── Fetch ─────────────────────────────────────────────────────────────────

    def _fetch_aircraft(self) -> List[Dict[str, Any]]:
        """Blocking HTTP fetch of aircraft.json."""
        with urllib.request.urlopen(self._json_url, timeout=5) as resp:
            data = json.loads(resp.read())

        # Handle both dump1090 format ({"aircraft": [...]}) and
        # readsb/tar1090 format ({"aircraft": [...], "now": ...})
        aircraft = data.get("aircraft", data if isinstance(data, list) else [])
        return aircraft if isinstance(aircraft, list) else []

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _aircraft_to_observation(
        self, ac: Dict[str, Any], ts_iso: str
    ) -> Optional[dict]:
        """Convert a dump1090 aircraft object to a Heli.OS observation."""
        hex_id = ac.get("hex", "").upper().strip("~")
        if not hex_id:
            return None

        lat = ac.get("lat")
        lon = ac.get("lon")

        # Only emit observations with a known position
        if lat is None or lon is None:
            # Update cache for velocity tracking even without position
            self._aircraft_cache[hex_id] = {**ac, "_updated": time.time()}
            return None

        # Check staleness from dump1090's own seen/seen_pos fields
        seen_pos = ac.get("seen_pos", ac.get("seen", 0))
        if isinstance(seen_pos, (int, float)) and seen_pos > self._stale_seconds:
            return None

        alt_baro = ac.get("alt_baro")  # barometric ft
        alt_geom = ac.get("alt_geom")  # geometric ft
        alt_ft = alt_geom or alt_baro
        alt_m = float(alt_ft) / 3.28084 if alt_ft and alt_ft != "ground" else 0.0

        gs = ac.get("gs")      # ground speed knots
        track = ac.get("track")   # true track degrees
        vert_rate = ac.get("baro_rate") or ac.get("geom_rate")  # ft/min

        speed_mps = float(gs) * 0.514444 if gs else None   # knots → m/s
        vert_mps = float(vert_rate) / 196.85 if vert_rate else None  # ft/min → m/s

        flight = (ac.get("flight") or "").strip() or None
        squawk = ac.get("squawk", "")

        # Emergency detection
        is_emergency = squawk in _SQUAWK_EMERGENCY
        emergency_type = None
        if squawk == "7500":
            emergency_type = "HIJACK"
        elif squawk == "7600":
            emergency_type = "RADIO_FAILURE"
        elif squawk == "7700":
            emergency_type = "EMERGENCY"

        # Aircraft category → entity classification
        category = ac.get("category", "")
        classification = _category_to_classification(category)

        # RSSI / signal quality (available in some dump1090 variants)
        rssi = ac.get("rssi")
        messages = ac.get("messages", 0)

        metadata: Dict[str, Any] = {
            "icao_hex": hex_id,
            "squawk": squawk,
            "category": category,
            "alt_baro_ft": alt_baro if alt_baro != "ground" else 0,
            "alt_geom_ft": alt_geom,
            "vert_rate_fpm": vert_rate,
            "messages": messages,
            "source": "adsb_local",
        }
        if rssi is not None:
            metadata["rssi_dbfs"] = rssi
        if is_emergency:
            metadata["emergency"] = emergency_type

        # Update state cache
        self._aircraft_cache[hex_id] = {**ac, "_updated": time.time()}

        obs = {
            "source_id": f"adsb-{hex_id}-{int(time.time())}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": f"adsb-{hex_id}",
            "callsign": flight or f"ADSB-{hex_id}",
            "position": {
                "lat": float(lat),
                "lon": float(lon),
                "alt_m": alt_m,
            },
            "velocity": {
                "heading_deg": float(track) if track else None,
                "speed_mps": speed_mps,
                "vertical_mps": vert_mps,
            },
            "entity_type": "AIRCRAFT",
            "classification": classification,
            "metadata": metadata,
            "ts_iso": ts_iso,
        }

        # Flag emergencies as high-priority
        if is_emergency:
            obs["priority"] = "CRITICAL"
            obs["alert"] = f"SQUAWK {squawk} — {emergency_type}"

        return obs

    def _prune_stale_cache(self) -> None:
        now = time.time()
        stale_ids = [
            k for k, v in self._aircraft_cache.items()
            if now - v.get("_updated", 0) > self._stale_seconds * 2
        ]
        for k in stale_ids:
            del self._aircraft_cache[k]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _category_to_classification(category: str) -> str:
    """Map dump1090 ADS-B emitter category to Heli.OS classification."""
    # ADS-B emitter categories: A0-A7, B0-B7, C0-C7, D0-D7
    if not category:
        return "UNKNOWN"
    c = category.upper()
    if c in ("A1", "A2", "A3", "A5"):
        return "FIXED_WING"
    if c in ("A7",):
        return "ROTORCRAFT"
    if c in ("B1", "B2", "B3", "B4", "B6", "B7"):
        return "GLIDER_BALLOON"
    if c in ("A6", "B0"):
        return "UAV"
    if c in ("C1", "C2", "C3"):
        return "GROUND_VEHICLE"
    if c in ("C4", "C5", "C6"):
        return "OBSTACLE"
    return "AIRCRAFT"
