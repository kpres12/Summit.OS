"""
Heli.OS — Space-Track.org Adapter
=====================================
Space-Track is the US Space Force / 18 SDS public catalog of all
unclassified Earth-orbiting objects. Provides TLEs, conjunction data
messages, decay predictions, etc.

Source:    https://www.space-track.org
License:   Free, requires registration.

Auth:
  Standard cookie-based login at /ajaxauth/login. Sessions last ~2 hours.

Environment / .env:
  SPACETRACK_USER
  SPACETRACK_PASS

Endpoints used:
  Login:     https://www.space-track.org/ajaxauth/login
  TLE feed:  https://www.space-track.org/basicspacedata/query/class/gp/...
  Decays:    https://www.space-track.org/basicspacedata/query/class/decay/...

Why this matters:
  - Space domain awareness — overhead pass forecasting, conjunction
    risk, RSO catalog
  - Pairs with CelesTrak adapter (already wired) for redundancy
  - Useful for federal customers who care about overhead schedule
    (CONUS overflight, ISR window planning, GPS interference checks)

Adapter modes (extra.mode):
  catalog     — periodic refresh of the active GP catalog (default,
                ~30k LEO objects)
  decays      — track recent decays (interesting orbits decaying soon)
  ondemand    — register-only

Register example:
  {
    "adapter_type": "space_track",
    "name": "Space-Track GP catalog",
    "poll_interval_seconds": 21600,    # every 6 hours
    "extra": {
      "mode": "catalog",
      "max_per_poll": 5000
    }
  }
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

try:
    import requests
except ImportError as e:
    raise ImportError("space_track_adapter requires `requests`") from e

from .base import BaseAdapter, AdapterConfig

logger = logging.getLogger("heli.adapters.spacetrack")


SPACETRACK_BASE = "https://www.space-track.org"


def _resolve_env(name: str) -> str:
    val = os.environ.get(name, "")
    if val:
        return val
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


class SpaceTrackClient:
    def __init__(self, user: Optional[str] = None,
                 password: Optional[str] = None):
        self._user = user or _resolve_env("SPACETRACK_USER")
        self._pass = password or _resolve_env("SPACETRACK_PASS")
        self._sess = requests.Session()
        self._sess.headers.update({"User-Agent": "Heli.OS/1.0"})
        self._logged_in = False

    def login(self) -> None:
        if self._logged_in:
            return
        if not self._user or not self._pass:
            raise RuntimeError(
                "SPACETRACK_USER / SPACETRACK_PASS not set — register at "
                "space-track.org and put them in .env")
        r = self._sess.post(f"{SPACETRACK_BASE}/ajaxauth/login",
                            data={"identity": self._user, "password": self._pass},
                            timeout=30)
        r.raise_for_status()
        self._logged_in = True
        logger.info("[spacetrack] logged in")

    def gp_catalog(self, limit: int = 5000,
                   format_: str = "json") -> list[dict]:
        """Active general-perturbations catalog — TLE/elements for ~30k LEO."""
        self.login()
        url = (f"{SPACETRACK_BASE}/basicspacedata/query/class/gp/"
               f"orderby/EPOCH%20desc/limit/{limit}/format/{format_}")
        r = self._sess.get(url, timeout=120)
        r.raise_for_status()
        return r.json() if format_ == "json" else r.text.splitlines()

    def decays(self, limit: int = 200) -> list[dict]:
        self.login()
        url = (f"{SPACETRACK_BASE}/basicspacedata/query/class/decay/"
               f"orderby/DECAY_EPOCH%20desc/limit/{limit}/format/json")
        r = self._sess.get(url, timeout=60)
        r.raise_for_status()
        return r.json()


class SpaceTrackAdapter(BaseAdapter):
    adapter_type = "space_track"

    def __init__(self, config: AdapterConfig, mqtt_client=None):
        super().__init__(config, mqtt_client=mqtt_client)
        extra = config.extra or {}
        self._mode = extra.get("mode", "catalog")
        self._max_per_poll = int(extra.get("max_per_poll", 5000))
        self._client = SpaceTrackClient()
        self._seen: set[str] = set()

    async def connect(self) -> None:
        if self._mode == "ondemand":
            return
        await asyncio.to_thread(self._client.login)

    async def disconnect(self) -> None:
        return

    async def stream_observations(self) -> AsyncIterator[dict]:
        if self._mode == "ondemand":
            while True:
                await asyncio.sleep(60)
            return
        while True:
            try:
                async for obs in self._poll():
                    yield obs
            except Exception as e:
                logger.warning("[spacetrack] %s poll error: %s",
                               self.config.adapter_id, e)
            await asyncio.sleep(self.config.poll_interval_seconds)

    async def _poll(self) -> AsyncIterator[dict]:
        if self._mode == "decays":
            rows = await asyncio.to_thread(self._client.decays, 200)
            for d in rows:
                key = str(d.get("NORAD_CAT_ID", ""))
                if not key or key in self._seen:
                    continue
                self._seen.add(key)
                yield {
                    "adapter_type":  self.adapter_type,
                    "adapter_id":    self.config.adapter_id,
                    "ts":            datetime.now(timezone.utc).isoformat(),
                    "entity_type":   "satellite",
                    "asset_type":    "decaying_object",
                    "norad_id":      key,
                    "object_name":   d.get("OBJECT_NAME"),
                    "country":       d.get("COUNTRY"),
                    "decay_epoch":   d.get("DECAY_EPOCH"),
                    "rcs_size":      d.get("RCS_SIZE"),
                }
        else:
            rows = await asyncio.to_thread(self._client.gp_catalog,
                                           self._max_per_poll, "json")
            n_new = 0
            for d in rows:
                key = str(d.get("NORAD_CAT_ID", ""))
                if not key:
                    continue
                # gp catalog is a snapshot — we emit the freshest TLE per cat ID per poll
                yield {
                    "adapter_type":  self.adapter_type,
                    "adapter_id":    self.config.adapter_id,
                    "ts":            datetime.now(timezone.utc).isoformat(),
                    "entity_type":   "satellite",
                    "asset_type":    "rso_tle",
                    "norad_id":      key,
                    "object_name":   d.get("OBJECT_NAME"),
                    "country":       d.get("COUNTRY"),
                    "epoch":         d.get("EPOCH"),
                    "mean_motion":   d.get("MEAN_MOTION"),
                    "eccentricity":  d.get("ECCENTRICITY"),
                    "inclination":   d.get("INCLINATION"),
                    "ra_of_asc_node": d.get("RA_OF_ASC_NODE"),
                    "arg_of_pericenter": d.get("ARG_OF_PERICENTER"),
                    "mean_anomaly":  d.get("MEAN_ANOMALY"),
                    "tle_line1":     d.get("TLE_LINE1"),
                    "tle_line2":     d.get("TLE_LINE2"),
                }
                n_new += 1
            logger.info("[spacetrack] %s -> %d gp records",
                        self.config.adapter_id, n_new)
