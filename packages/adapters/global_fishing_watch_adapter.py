"""
Heli.OS — Global Fishing Watch (GFW) v3 API Adapter
=======================================================
Global Fishing Watch is a free public service (Oceana, Google, SkyTruth)
that publishes vessel tracking data, fishing activity inferences, and
*dark vessel detection* (radar + optical SAR analysis) via a documented
JSON REST API.

Source:    https://globalfishingwatch.org/our-apis/
License:   Free for non-commercial / research use; commercial use
           requires a separate GFW agreement.

Auth:
  GFW API tokens are issued through the GFW developer portal (free
  registration). Tokens are bearer tokens with no rotation requirement.

Environment / .env:
  GFW_API_TOKEN

Endpoints used:
  Events (encounters, port-visits, loitering, gaps, fishing):
    https://gateway.api.globalfishingwatch.org/v3/events
  4Wings vessel-presence aggregations (heatmaps, dark-vessel mosaics):
    https://gateway.api.globalfishingwatch.org/v3/4wings/...
  Vessel search / details:
    https://gateway.api.globalfishingwatch.org/v3/vessels/search

Adapter modes (extra.mode):
  events_aoi  — periodic /v3/events search over an AOI; emit one
                observation per new event (default)
  ondemand    — register-only; no scheduled polling

Why this matters:
  - Replaces the xView3 dark-vessel data we couldn't get programmatically
  - Real federal MDA / IUU-fishing telemetry, with actual labels
  - Same data Maritime Awareness Centers and US Coast Guard already use

Register example:
  {
    "adapter_type": "global_fishing_watch",
    "name": "GFW — Gulf of Mexico events",
    "poll_interval_seconds": 1800,
    "extra": {
      "mode": "events_aoi",
      "datasets": ["public-global-fishing-events:latest"],
      "bbox": [-97.5, 24.0, -82.0, 31.0],
      "max_age_days": 7,
      "limit_per_poll": 100
    }
  }
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

try:
    import requests
except ImportError as e:
    raise ImportError("global_fishing_watch_adapter requires `requests`") from e

from .base import BaseAdapter, AdapterConfig

logger = logging.getLogger("heli.adapters.gfw")


GFW_BASE = "https://gateway.api.globalfishingwatch.org/v3"

# Default datasets — public for free-tier accounts
DEFAULT_DATASETS = [
    "public-global-fishing-events:latest",
    "public-global-encounters-events:latest",
    "public-global-port-visits-events:latest",
    "public-global-loitering-events:latest",
    "public-global-gaps-events:latest",   # AIS gap = candidate dark vessel
]


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


class GFWClient:
    """Synchronous client for events / vessels / 4wings endpoints."""

    def __init__(self, token: Optional[str] = None):
        self._token = token or _resolve_env("GFW_API_TOKEN")

    def _headers(self) -> dict:
        if not self._token:
            raise RuntimeError(
                "GFW_API_TOKEN not set — register at "
                "https://globalfishingwatch.org/our-apis/ for a free token "
                "and put it in .env")
        return {"Authorization": f"Bearer {self._token}",
                "User-Agent": "Heli.OS/1.0"}

    def search_events(self, datasets: list[str], bbox: list[float],
                      start: datetime, end: datetime,
                      limit: int = 100) -> list[dict]:
        body = {
            "datasets": datasets,
            "startDate": start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "endDate":   end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [bbox[0], bbox[1]], [bbox[2], bbox[1]],
                    [bbox[2], bbox[3]], [bbox[0], bbox[3]],
                    [bbox[0], bbox[1]],
                ]],
            },
            "limit": limit,
        }
        r = requests.post(f"{GFW_BASE}/events", json=body,
                          headers=self._headers(), timeout=60)
        r.raise_for_status()
        return r.json().get("entries") or r.json().get("data") or []

    def vessel_search(self, query: str, datasets: list[str],
                      limit: int = 25) -> list[dict]:
        params = {
            "query": query,
            "datasets[0]": datasets[0] if datasets else "public-global-vessels:latest",
            "limit": limit,
        }
        r = requests.get(f"{GFW_BASE}/vessels/search", params=params,
                         headers=self._headers(), timeout=60)
        r.raise_for_status()
        return r.json().get("entries") or r.json().get("data") or []


class GlobalFishingWatchAdapter(BaseAdapter):
    adapter_type = "global_fishing_watch"

    def __init__(self, config: AdapterConfig, mqtt_client=None):
        super().__init__(config, mqtt_client=mqtt_client)
        extra = config.extra or {}
        self._mode = extra.get("mode", "events_aoi")
        self._datasets: list[str] = list(extra.get("datasets", DEFAULT_DATASETS))
        self._bbox: list[float] = list(extra.get("bbox", [-180, -90, 180, 90]))
        self._max_age_days: int = int(extra.get("max_age_days", 7))
        self._limit: int = int(extra.get("limit_per_poll", 100))
        self._client = GFWClient()
        self._seen: set[str] = set()

    async def connect(self) -> None:
        if self._mode == "ondemand":
            return
        # Touch the token by attempting a tiny request
        if not _resolve_env("GFW_API_TOKEN"):
            raise RuntimeError("GFW_API_TOKEN not configured")

    async def disconnect(self) -> None:
        return

    async def stream_observations(self) -> AsyncIterator[dict]:
        if self._mode != "events_aoi":
            while True:
                await asyncio.sleep(60)
            return
        while True:
            try:
                async for obs in self._poll_events():
                    yield obs
            except Exception as e:
                logger.warning("[GFW] %s poll error: %s",
                               self.config.adapter_id, e)
            await asyncio.sleep(self.config.poll_interval_seconds)

    async def _poll_events(self) -> AsyncIterator[dict]:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=self._max_age_days)

        def _q():
            return self._client.search_events(
                self._datasets, self._bbox, start, end, limit=self._limit)
        events = await asyncio.to_thread(_q)
        n_new = 0
        for e in events:
            ev_id = e.get("id") or f"{e.get('vessel', {}).get('id', '')}-{e.get('start', '')}"
            if not ev_id or ev_id in self._seen:
                continue
            self._seen.add(ev_id)
            n_new += 1
            yield self._event_to_obs(e)
        if len(self._seen) > 50_000:
            self._seen = set(list(self._seen)[-25_000:])
        logger.info("[GFW] %s -> %d new events", self.config.adapter_id, n_new)

    def _event_to_obs(self, e: dict) -> dict:
        pos = e.get("position") or {}
        lat = float(pos.get("lat", 0)) if pos else 0.0
        lon = float(pos.get("lon", 0)) if pos else 0.0
        v = e.get("vessel") or {}
        return {
            "adapter_type":  self.adapter_type,
            "adapter_id":    self.config.adapter_id,
            "ts":            datetime.now(timezone.utc).isoformat(),
            "entity_type":   "event_observation",
            "asset_type":    "vessel_event",
            "event_id":      e.get("id"),
            "event_type":    e.get("type"),
            "start":         e.get("start"),
            "end":           e.get("end"),
            "lat":           lat,
            "lon":           lon,
            "vessel_id":     v.get("id"),
            "vessel_name":   v.get("name"),
            "vessel_flag":   v.get("flag"),
            "vessel_class":  v.get("type"),
            "fishing_hours": (e.get("fishing") or {}).get("totalHours"),
            "is_dark":       e.get("type") == "gap",  # AIS gap = candidate dark
        }
