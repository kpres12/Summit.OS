"""
Heli.OS — Microsoft Planetary Computer Adapter
==================================================
Microsoft's free public STAC catalog hosting Sentinel-1, Sentinel-2,
Landsat, MODIS, NAIP, Copernicus DEM, and many other Earth-observation
collections. **No registration required for STAC search.** Asset
download URLs are signed via the public SAS-issuance API (no auth
required either, just a single REST call per asset).

Source:    https://planetarycomputer.microsoft.com/api/stac/v1/
SAS API:   https://planetarycomputer.microsoft.com/api/sas/v1/sign
License:   Free, no auth needed. Microsoft hosts pursuant to NASA /
           ESA / USGS data-sharing agreements.

This is the third programmatic alternative for Sentinel-1/2/3 imagery
alongside the Copernicus Data Space (CDSE) adapter and the paid Sentinel
Hub adapter, useful for redundancy and for Microsoft Azure-aligned
deployments.

Adapter modes (extra.mode):
  catalog_aoi  — periodic STAC search over an AOI (default)
  ondemand     — register-only

Config:
  collection         — STAC collection id (e.g. "sentinel-2-l2a", "landsat-c2-l2",
                       "naip", "sentinel-1-grd", "modis-09A1-061")
  bbox               — [min_lon, min_lat, max_lon, max_lat]
  max_age_days       — recency window
  limit_per_poll     — max scenes per poll
  max_cloud_cover    — Sentinel-2 / Landsat only

Register example:
  {
    "adapter_type": "planetary_computer",
    "name": "PC — Sentinel-2 GoM",
    "poll_interval_seconds": 3600,
    "extra": {
      "collection": "sentinel-2-l2a",
      "bbox": [-97.5, 24.0, -82.0, 31.0],
      "max_age_days": 7,
      "max_cloud_cover": 30
    }
  }
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Optional

try:
    import requests
except ImportError as e:
    raise ImportError("planetary_computer_adapter requires `requests`") from e

from .base import BaseAdapter, AdapterConfig

logger = logging.getLogger("heli.adapters.planetary_computer")


PC_STAC_BASE = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_SAS_BASE  = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"


class PlanetaryComputerClient:
    """Synchronous client for STAC search + SAS-token asset signing.

    Usable directly from training pipelines:

        from packages.adapters.planetary_computer_adapter import PlanetaryComputerClient
        pc = PlanetaryComputerClient()
        items = pc.catalog_search("sentinel-2-l2a",
                                   bbox=[-97.5,24,-82,31],
                                   datetime_range="2024-09-01/2024-09-15",
                                   max_cloud_cover=30, limit=50)
        for item in items:
            for asset_name, asset in (item.get("assets") or {}).items():
                signed_url = pc.sign(asset["href"])
                # ... fetch signed_url with requests ...
    """

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "Heli.OS/1.0"})

    def catalog_search(self, collection: str, bbox: list[float],
                       datetime_range: str, limit: int = 100,
                       max_cloud_cover: Optional[float] = None) -> list[dict]:
        body: dict = {
            "bbox": bbox,
            "datetime": datetime_range,
            "collections": [collection],
            "limit": limit,
        }
        if max_cloud_cover is not None and ("sentinel-2" in collection or "landsat" in collection):
            body["filter-lang"] = "cql2-json"
            body["filter"] = {
                "op": "<=",
                "args": [{"property": "eo:cloud_cover"}, float(max_cloud_cover)],
            }
        items: list[dict] = []
        url = f"{PC_STAC_BASE}/search"
        for _ in range(20):
            r = self._session.post(url, json=body, timeout=60)
            if r.status_code == 429:
                time.sleep(5)
                continue
            r.raise_for_status()
            data = r.json()
            items.extend(data.get("features") or [])
            links = data.get("links") or []
            nxt = next((l for l in links if l.get("rel") == "next"), None)
            if not nxt:
                break
            url = nxt.get("href")
            body = nxt.get("body") or body
            if not url:
                break
        logger.info("[PC] %s %s -> %d scenes", collection, bbox, len(items))
        return items

    def sign(self, href: str) -> str:
        """Sign an asset URL with a SAS token. Returns the signed URL."""
        r = self._session.get(PC_SAS_BASE, params={"href": href}, timeout=30)
        r.raise_for_status()
        return r.json().get("href", href)


class PlanetaryComputerAdapter(BaseAdapter):
    adapter_type = "planetary_computer"

    def __init__(self, config: AdapterConfig, mqtt_client=None):
        super().__init__(config, mqtt_client=mqtt_client)
        extra = config.extra or {}
        self._collection: str = extra.get("collection", "sentinel-2-l2a")
        self._bbox: list[float] = list(extra.get("bbox", [-125, 24, -66, 49]))
        self._max_age_days: int = int(extra.get("max_age_days", 7))
        self._limit: int = int(extra.get("limit_per_poll", 50))
        self._max_cc = extra.get("max_cloud_cover")
        self._client = PlanetaryComputerClient()
        self._seen: set[str] = set()

    async def connect(self) -> None:
        return

    async def disconnect(self) -> None:
        return

    async def stream_observations(self) -> AsyncIterator[dict]:
        while True:
            try:
                async for obs in self._poll():
                    yield obs
            except Exception as e:
                logger.warning("[PC] %s poll error: %s",
                               self.config.adapter_id, e)
            await asyncio.sleep(self.config.poll_interval_seconds)

    async def _poll(self) -> AsyncIterator[dict]:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=self._max_age_days)
        rng = f"{start.isoformat()}/{end.isoformat()}"

        def _q():
            return self._client.catalog_search(
                self._collection, self._bbox, rng, self._limit,
                float(self._max_cc) if self._max_cc is not None else None)
        items = await asyncio.to_thread(_q)

        n_new = 0
        for item in items:
            sid = item.get("id")
            if not sid or sid in self._seen:
                continue
            self._seen.add(sid)
            n_new += 1
            yield self._item_to_obs(item)
        if len(self._seen) > 50_000:
            self._seen = set(list(self._seen)[-25_000:])
        logger.info("[PC] %s -> %d new scenes", self.config.adapter_id, n_new)

    def _item_to_obs(self, item: dict) -> dict:
        props = item.get("properties") or {}
        geom = item.get("geometry") or {}
        cx = cy = 0.0
        try:
            ring = (geom.get("coordinates") or [])[0]
            xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
            cx = sum(xs) / len(xs); cy = sum(ys) / len(ys)
        except Exception:
            pass
        return {
            "adapter_type": self.adapter_type,
            "adapter_id":   self.config.adapter_id,
            "ts":           datetime.now(timezone.utc).isoformat(),
            "entity_type":  "sensor_observation",
            "asset_type":   "satellite_scene",
            "collection":   self._collection,
            "scene_id":     item.get("id"),
            "datetime":     props.get("datetime"),
            "lat": cy, "lon": cx,
            "geometry":     geom,
            "cloud_cover":  props.get("eo:cloud_cover"),
            "platform":     props.get("platform") or props.get("constellation"),
            "asset_keys":   list((item.get("assets") or {}).keys()),
            "metadata":     {k: v for k, v in props.items() if k != "datetime"},
        }
