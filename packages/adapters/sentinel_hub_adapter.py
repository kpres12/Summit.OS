"""
Heli.OS — Sentinel Hub Adapter
=================================
Pulls Earth observation imagery (Sentinel-1 SAR, Sentinel-2 optical,
Sentinel-3, Landsat 8/9, MODIS, DEM) from the Sentinel Hub Process API
and STAC Catalog. Used by:

  - Vision model training pipelines (vessel detection CNN, burn-scar
    U-Net, smoke plume CNN, oil-spill SAR detection, flood inundation
    U-Net, urban thermal heat-island)
  - On-demand imagery requests from operators (mission overlays)
  - Periodic AOI scans for change detection (e.g., post-fire BDA)

Auth:
  OAuth2 client_credentials flow against
  https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token
  Tokens are cached in-memory and refreshed ~60s before expiry.

Environment / .env:
  SH_CLIENT_ID
  SH_CLIENT_SECRET

Adapter modes (AdapterConfig.extra.mode):
  catalog_aoi  — periodic STAC search over an AOI; emit observation per scene
  process_aoi  — periodic Process API call; emit a fetched raster as observation
  ondemand     — no scheduled polling; only respond to send_command("FETCH", ...)

Register example:
  {
    "adapter_type": "sentinel_hub",
    "name": "Sentinel-1 SAR — Gulf of Mexico",
    "poll_interval_seconds": 3600,
    "extra": {
      "mode": "catalog_aoi",
      "collection": "sentinel-1-grd",
      "bbox": [-97.5, 24.0, -82.0, 31.0],
      "max_cloud_cover": null,
      "max_age_days": 7,
      "limit_per_poll": 20
    }
  }

License:
  Sentinel Hub commercial — uses your processing-unit (PU) quota. Free
  trial accounts work; tracking PU consumption is on the operator side.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

try:
    import requests
except ImportError as e:
    raise ImportError("sentinel_hub_adapter requires `requests` (pip install requests)") from e

from .base import BaseAdapter, AdapterConfig

logger = logging.getLogger("heli.adapters.sentinel_hub")


SH_API_BASE   = "https://services.sentinel-hub.com"
SH_TOKEN_URL  = f"{SH_API_BASE}/auth/realms/main/protocol/openid-connect/token"
SH_PROCESS_URL = f"{SH_API_BASE}/api/v1/process"
SH_CATALOG_BASE = f"{SH_API_BASE}/api/v1/catalog/1.0.0"
SH_STATS_URL  = f"{SH_API_BASE}/api/v1/statistics"


# Map common short names to STAC collection ids
_COLLECTION_ALIASES = {
    "sentinel-1":           "sentinel-1-grd",
    "sentinel-1-grd":       "sentinel-1-grd",
    "sentinel-2":           "sentinel-2-l2a",
    "sentinel-2-l1c":       "sentinel-2-l1c",
    "sentinel-2-l2a":       "sentinel-2-l2a",
    "sentinel-3":           "sentinel-3-olci",
    "sentinel-3-olci":      "sentinel-3-olci",
    "sentinel-3-slstr":     "sentinel-3-slstr",
    "landsat":              "landsat-ot-l2",
    "landsat-ot-l2":        "landsat-ot-l2",
    "landsat-tm-l2":        "landsat-tm-l2",
    "modis":                "modis",
    "dem":                  "dem",
}


def _resolve_env(name: str) -> str:
    """Read env var, falling back to project .env file (no leakage to logs)."""
    val = os.environ.get(name, "")
    if val:
        return val
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


class _TokenManager:
    """OAuth2 client_credentials with in-memory caching + ~60s pre-expiry refresh."""

    def __init__(self, client_id: str, client_secret: str):
        self._client_id = client_id
        self._client_secret = client_secret
        self._token: Optional[str] = None
        self._expires_at: float = 0.0
        self._lock = asyncio.Lock()

    async def get_token(self) -> str:
        async with self._lock:
            if self._token and time.time() < self._expires_at - 60:
                return self._token
            return await asyncio.to_thread(self._refresh)

    def _refresh(self) -> str:
        if not self._client_id or not self._client_secret:
            raise RuntimeError(
                "SH_CLIENT_ID / SH_CLIENT_SECRET not set — register at "
                "https://www.sentinel-hub.com/develop/api/ and put them in .env")
        resp = requests.post(SH_TOKEN_URL, data={
            "grant_type":    "client_credentials",
            "client_id":     self._client_id,
            "client_secret": self._client_secret,
        }, timeout=30, headers={"User-Agent": "Heli.OS/1.0"})
        resp.raise_for_status()
        body = resp.json()
        self._token = body["access_token"]
        self._expires_at = time.time() + int(body.get("expires_in", 3600))
        logger.info("[SentinelHub] auth: token refreshed (expires in %ds)",
                    int(self._expires_at - time.time()))
        return self._token


# ---------------------------------------------------------------------------
# Public client (usable outside the adapter framework, e.g. from training scripts)
# ---------------------------------------------------------------------------


class SentinelHubClient:
    """Thin synchronous client for Sentinel Hub Process / Catalog / Statistics APIs.

    Usable directly from training pipelines:

        from packages.adapters.sentinel_hub_adapter import SentinelHubClient
        sh = SentinelHubClient()
        scenes = sh.catalog_search(
            collection="sentinel-1-grd",
            bbox=[-97.5, 24.0, -82.0, 31.0],
            datetime_range="2024-09-01T00:00:00Z/2024-09-15T23:59:59Z",
            limit=50,
        )
    """

    def __init__(self, client_id: Optional[str] = None,
                 client_secret: Optional[str] = None):
        cid = client_id or _resolve_env("SH_CLIENT_ID")
        sec = client_secret or _resolve_env("SH_CLIENT_SECRET")
        self._tokens = _TokenManager(cid, sec)

    def _headers(self) -> dict:
        # Synchronous wrapper around the async token cache for ease of use
        loop = asyncio.new_event_loop()
        try:
            token = loop.run_until_complete(self._tokens.get_token())
        finally:
            loop.close()
        return {
            "Authorization": f"Bearer {token}",
            "User-Agent": "Heli.OS/1.0",
        }

    def catalog_search(self, collection: str, bbox: list[float],
                       datetime_range: str, limit: int = 100,
                       max_cloud_cover: Optional[float] = None) -> list[dict]:
        """STAC ItemSearch — returns a list of scene metadata items."""
        col = _COLLECTION_ALIASES.get(collection.lower(), collection)
        body: dict = {
            "bbox": bbox,
            "datetime": datetime_range,
            "collections": [col],
            "limit": limit,
        }
        if max_cloud_cover is not None and "sentinel-2" in col:
            body["filter-lang"] = "cql2-json"
            body["filter"] = {
                "op": "<=",
                "args": [{"property": "eo:cloud_cover"}, max_cloud_cover],
            }
        all_features: list[dict] = []
        next_url = f"{SH_CATALOG_BASE}/search"
        for _ in range(20):  # cap at 20 pages
            r = requests.post(next_url, json=body, headers=self._headers(),
                              timeout=60)
            if r.status_code == 429:
                logger.warning("[SentinelHub] catalog 429 — sleeping 10s")
                time.sleep(10)
                continue
            r.raise_for_status()
            payload = r.json()
            features = payload.get("features") or []
            all_features.extend(features)
            links = payload.get("links") or []
            next_link = next((l for l in links if l.get("rel") == "next"), None)
            if not next_link:
                break
            next_url = next_link.get("href")
            body = next_link.get("body") or body
            if not next_url:
                break
        logger.info("[SentinelHub] catalog: %s %s -> %d scenes", col, bbox, len(all_features))
        return all_features

    def process(self, evalscript: str, input_collection: str, bbox: list[float],
                width: int, height: int, datetime_range: str,
                output_format: str = "image/tiff") -> bytes:
        """Process API — fetch a raster for an AOI/time as bytes (TIFF/PNG/JPEG).

        evalscript: see https://docs.sentinel-hub.com/api/latest/evalscript/v3/
        """
        col = _COLLECTION_ALIASES.get(input_collection.lower(), input_collection)
        payload = {
            "input": {
                "bounds": {"bbox": bbox, "properties": {"crs":
                            "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}},
                "data": [{
                    "type": col,
                    "dataFilter": {"timeRange": {
                        "from": datetime_range.split("/")[0],
                        "to":   datetime_range.split("/")[1],
                    }},
                }],
            },
            "output": {
                "width": width,
                "height": height,
                "responses": [{
                    "identifier": "default",
                    "format": {"type": output_format},
                }],
            },
            "evalscript": evalscript,
        }
        r = requests.post(SH_PROCESS_URL, json=payload, headers=self._headers(),
                          timeout=180)
        if r.status_code == 429:
            logger.warning("[SentinelHub] process 429 — sleeping 10s")
            time.sleep(10)
            r = requests.post(SH_PROCESS_URL, json=payload, headers=self._headers(),
                              timeout=180)
        r.raise_for_status()
        return r.content

    def statistics(self, evalscript: str, input_collection: str, geometry: dict,
                   datetime_range: str, aggregation_interval: str = "P1D") -> dict:
        """Statistical API — zonal stats over a polygon for a date range."""
        col = _COLLECTION_ALIASES.get(input_collection.lower(), input_collection)
        payload = {
            "input": {
                "bounds": {"geometry": geometry, "properties": {"crs":
                            "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}},
                "data": [{"type": col}],
            },
            "aggregation": {
                "timeRange": {
                    "from": datetime_range.split("/")[0],
                    "to":   datetime_range.split("/")[1],
                },
                "aggregationInterval": {"of": aggregation_interval},
                "evalscript": evalscript,
            },
        }
        r = requests.post(SH_STATS_URL, json=payload, headers=self._headers(),
                          timeout=180)
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Streaming adapter — registered via packages.adapters.registry
# ---------------------------------------------------------------------------


class SentinelHubAdapter(BaseAdapter):
    """Stream Sentinel Hub scene metadata (catalog mode) into Heli.OS as
    observations. Each observation describes one new scene over the AOI."""

    adapter_type = "sentinel_hub"

    def __init__(self, config: AdapterConfig, mqtt_client=None):
        super().__init__(config, mqtt_client=mqtt_client)
        extra = config.extra or {}
        self._mode: str = extra.get("mode", "catalog_aoi")
        self._collection: str = extra.get("collection", "sentinel-1-grd")
        self._bbox: list[float] = list(extra.get("bbox", [-125, 24, -66, 49]))
        self._max_age_days: int = int(extra.get("max_age_days", 7))
        self._limit_per_poll: int = int(extra.get("limit_per_poll", 50))
        self._max_cloud_cover = extra.get("max_cloud_cover")  # may be None

        cid = _resolve_env("SH_CLIENT_ID")
        sec = _resolve_env("SH_CLIENT_SECRET")
        self._tokens = _TokenManager(cid, sec)
        self._seen_scene_ids: set[str] = set()

    async def connect(self) -> None:
        # Force a token fetch so we fail fast on bad creds
        if self._mode == "ondemand":
            logger.info("[SentinelHub] %s: ondemand mode — no scheduled polling",
                        self.config.adapter_id)
            return
        try:
            await self._tokens.get_token()
        except Exception as e:
            logger.error("[SentinelHub] %s: auth failed at connect: %s",
                         self.config.adapter_id, e)
            raise

    async def disconnect(self) -> None:
        return

    async def stream_observations(self) -> AsyncIterator[dict]:
        if self._mode != "catalog_aoi":
            # ondemand mode: never yield anything; framework keeps adapter
            # registered so send_command() works against it
            while True:
                await asyncio.sleep(60)
            return

        while True:
            try:
                async for obs in self._poll_catalog():
                    yield obs
            except Exception as e:
                logger.warning("[SentinelHub] %s poll error: %s",
                               self.config.adapter_id, e)
            await asyncio.sleep(self.config.poll_interval_seconds)

    async def _poll_catalog(self) -> AsyncIterator[dict]:
        """One pass of STAC search — yield each new scene as an observation."""
        token = await self._tokens.get_token()
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=self._max_age_days)
        col = _COLLECTION_ALIASES.get(self._collection.lower(), self._collection)
        body: dict = {
            "bbox": self._bbox,
            "datetime": f"{start.isoformat()}/{end.isoformat()}",
            "collections": [col],
            "limit": self._limit_per_poll,
        }
        if self._max_cloud_cover is not None and "sentinel-2" in col:
            body["filter-lang"] = "cql2-json"
            body["filter"] = {
                "op": "<=",
                "args": [{"property": "eo:cloud_cover"},
                         float(self._max_cloud_cover)],
            }

        def _post():
            return requests.post(f"{SH_CATALOG_BASE}/search", json=body,
                                 headers={"Authorization": f"Bearer {token}",
                                          "User-Agent": "Heli.OS/1.0"},
                                 timeout=60)

        r = await asyncio.to_thread(_post)
        r.raise_for_status()
        features = r.json().get("features") or []
        new_features = [f for f in features if f.get("id") not in self._seen_scene_ids]
        for f in new_features:
            self._seen_scene_ids.add(f.get("id"))
            yield self._scene_to_observation(f, col)

    def _scene_to_observation(self, feat: dict, collection: str) -> dict:
        props = feat.get("properties") or {}
        geom = feat.get("geometry") or {}
        # Compute centroid for entity placement
        cx = cy = 0.0
        try:
            coords = geom.get("coordinates") or []
            if geom.get("type") == "Polygon" and coords:
                ring = coords[0]
                xs = [p[0] for p in ring]
                ys = [p[1] for p in ring]
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)
        except Exception:
            pass

        return {
            "adapter_type":  self.adapter_type,
            "adapter_id":    self.config.adapter_id,
            "ts":            datetime.now(timezone.utc).isoformat(),
            "entity_type":   "sensor_observation",
            "asset_type":    "satellite_scene",
            "collection":    collection,
            "scene_id":      feat.get("id"),
            "datetime":      props.get("datetime"),
            "lat":           cy,
            "lon":           cx,
            "geometry":      geom,
            "cloud_cover":   props.get("eo:cloud_cover"),
            "platform":      props.get("platform") or props.get("constellation"),
            "metadata":      {k: v for k, v in props.items()
                              if k not in {"datetime"}},
        }

    async def send_command(self, command: str, params: dict | None = None) -> dict:
        """Operator/mission-driven on-demand fetch.

        Commands:
          FETCH_RASTER   — params: collection, bbox, width, height, datetime_range,
                                   evalscript, output_format
                          returns base64-encoded raster bytes
          CATALOG_SEARCH — params: collection, bbox, datetime_range, limit,
                                   max_cloud_cover
                          returns list of STAC features
          STATISTICS     — params: collection, geometry, datetime_range,
                                   evalscript, aggregation_interval
                          returns Statistics API response
        """
        params = params or {}
        cmd = command.upper()
        cli = SentinelHubClient()  # uses same env-loaded creds
        if cmd == "CATALOG_SEARCH":
            features = await asyncio.to_thread(cli.catalog_search,
                params.get("collection", "sentinel-1-grd"),
                params.get("bbox", self._bbox),
                params.get("datetime_range", ""),
                int(params.get("limit", 50)),
                params.get("max_cloud_cover"),
            )
            return {"status": "ok", "feature_count": len(features),
                    "features": features}
        if cmd == "FETCH_RASTER":
            import base64
            raster = await asyncio.to_thread(cli.process,
                params["evalscript"],
                params.get("collection", "sentinel-1-grd"),
                params.get("bbox", self._bbox),
                int(params.get("width", 512)),
                int(params.get("height", 512)),
                params["datetime_range"],
                params.get("output_format", "image/tiff"),
            )
            return {"status": "ok",
                    "format": params.get("output_format", "image/tiff"),
                    "bytes_b64": base64.b64encode(raster).decode("ascii"),
                    "size_bytes": len(raster)}
        if cmd == "STATISTICS":
            stats = await asyncio.to_thread(cli.statistics,
                params["evalscript"],
                params.get("collection", "sentinel-2-l2a"),
                params["geometry"],
                params["datetime_range"],
                params.get("aggregation_interval", "P1D"),
            )
            return {"status": "ok", "statistics": stats}
        return {"status": "error", "message": f"unknown command: {command}"}


# Auto-registration is handled by adapters.registry._try_register_builtins
# which adds ("adapters.sentinel_hub_adapter", "SentinelHubAdapter") to its
# import list. No self-registration needed here.
