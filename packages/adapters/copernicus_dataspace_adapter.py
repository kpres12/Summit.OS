"""
Heli.OS — Copernicus Data Space Ecosystem (CDSE) Adapter
============================================================
Free public alternative to commercial Sentinel Hub. Run by ESA / the EU
under the Copernicus programme. Same Sentinel-1 SAR / Sentinel-2 optical
/ Sentinel-3 / Sentinel-5P data; same STAC + OData + Process API patterns;
free quota covers nearly all civilian + federal mission-rehearsal use.

Auth:
  OAuth2 password (resource owner) flow against
  https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token
  (registration is free at https://dataspace.copernicus.eu — single
  account → username + password → no client-secret OAuth dance required).

Environment / .env:
  CDSE_USERNAME
  CDSE_PASSWORD

Endpoints used:
  STAC catalog:    https://catalogue.dataspace.copernicus.eu/stac/v1/
  OData (legacy):  https://catalogue.dataspace.copernicus.eu/odata/v1/
  Sentinel Hub-compatible Process API: https://sh.dataspace.copernicus.eu

Why this matters:
  Replaces the paid Sentinel Hub adapter for projects that don't need
  enterprise SLAs. Frees up the deferred Sentinel Hub creds problem.

Usage:
    from packages.adapters.copernicus_dataspace_adapter import (
        CDSEClient, CopernicusDataSpaceAdapter,
    )
    cli = CDSEClient()
    scenes = cli.catalog_search(
        collection="SENTINEL-1",
        bbox=[-97.5, 24.0, -82.0, 31.0],
        datetime_range="2024-09-01T00:00:00Z/2024-09-15T23:59:59Z",
        limit=50,
    )
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

try:
    import requests
except ImportError as e:
    raise ImportError("copernicus_dataspace_adapter requires `requests`") from e

from .base import BaseAdapter, AdapterConfig

logger = logging.getLogger("heli.adapters.cdse")


CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_STAC_BASE = "https://catalogue.dataspace.copernicus.eu/stac/v1"
CDSE_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
CDSE_ODATA_BASE = "https://catalogue.dataspace.copernicus.eu/odata/v1"


_COLLECTION_ALIASES = {
    "sentinel-1":     "SENTINEL-1",
    "sentinel-1-grd": "SENTINEL-1",
    "sentinel-1-slc": "SENTINEL-1",
    "sentinel-2":     "SENTINEL-2",
    "sentinel-2-l1c": "SENTINEL-2",
    "sentinel-2-l2a": "SENTINEL-2",
    "sentinel-3":     "SENTINEL-3",
    "sentinel-5p":    "SENTINEL-5P",
}


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


class _CDSETokenManager:
    """OAuth2 password-flow token cache with ~60s pre-expiry refresh."""

    def __init__(self, username: str, password: str):
        self._username = username
        self._password = password
        self._token: Optional[str] = None
        self._expires_at: float = 0.0
        self._lock = asyncio.Lock()

    async def get_token(self) -> str:
        async with self._lock:
            if self._token and time.time() < self._expires_at - 60:
                return self._token
            return await asyncio.to_thread(self._refresh)

    def _refresh(self) -> str:
        if not self._username or not self._password:
            raise RuntimeError(
                "CDSE_USERNAME / CDSE_PASSWORD not set — register at "
                "https://dataspace.copernicus.eu and put them in .env")
        resp = requests.post(CDSE_TOKEN_URL, data={
            "grant_type":    "password",
            "client_id":     "cdse-public",
            "username":      self._username,
            "password":      self._password,
        }, timeout=30, headers={"User-Agent": "Heli.OS/1.0"})
        resp.raise_for_status()
        body = resp.json()
        self._token = body["access_token"]
        self._expires_at = time.time() + int(body.get("expires_in", 600))
        logger.info("[CDSE] token refreshed (expires in %ds)",
                    int(self._expires_at - time.time()))
        return self._token


class CDSEClient:
    """Synchronous client. Usable from training pipelines."""

    def __init__(self, username: Optional[str] = None,
                 password: Optional[str] = None):
        u = username or _resolve_env("CDSE_USERNAME")
        p = password or _resolve_env("CDSE_PASSWORD")
        self._tokens = _CDSETokenManager(u, p)

    def _headers(self) -> dict:
        loop = asyncio.new_event_loop()
        try:
            token = loop.run_until_complete(self._tokens.get_token())
        finally:
            loop.close()
        return {"Authorization": f"Bearer {token}",
                "User-Agent": "Heli.OS/1.0"}

    def catalog_search(self, collection: str, bbox: list[float],
                       datetime_range: str, limit: int = 100,
                       max_cloud_cover: Optional[float] = None) -> list[dict]:
        col = _COLLECTION_ALIASES.get(collection.lower(), collection)
        body: dict = {
            "bbox": bbox,
            "datetime": datetime_range,
            "collections": [col],
            "limit": limit,
        }
        if max_cloud_cover is not None and "SENTINEL-2" in col.upper():
            body["filter-lang"] = "cql2-json"
            body["filter"] = {
                "op": "<=",
                "args": [{"property": "eo:cloud_cover"}, max_cloud_cover],
            }
        all_features: list[dict] = []
        url = f"{CDSE_STAC_BASE}/search"
        for _ in range(20):
            r = requests.post(url, json=body, headers=self._headers(), timeout=60)
            if r.status_code == 429:
                time.sleep(5)
                continue
            r.raise_for_status()
            data = r.json()
            all_features.extend(data.get("features") or [])
            links = data.get("links") or []
            nxt = next((l for l in links if l.get("rel") == "next"), None)
            if not nxt:
                break
            url = nxt.get("href")
            body = nxt.get("body") or body
            if not url:
                break
        logger.info("[CDSE] catalog %s %s -> %d scenes",
                    col, bbox, len(all_features))
        return all_features

    def odata_query(self, query: str, top: int = 100) -> list[dict]:
        """Legacy OData query — useful for advanced filtering (orbit number,
        platform, processing baseline, etc.) that STAC doesn't expose."""
        url = f"{CDSE_ODATA_BASE}/Products?$filter={query}&$top={top}"
        r = requests.get(url, headers=self._headers(), timeout=60)
        r.raise_for_status()
        return r.json().get("value") or []

    def download_product(self, product_id: str, out_path: Path) -> Path:
        """Download a Sentinel product by OData ID. Returns the local path."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{CDSE_ODATA_BASE}/Products({product_id})/$value"
        with requests.get(url, headers=self._headers(), stream=True,
                          timeout=600) as r:
            r.raise_for_status()
            with out_path.open("wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
        logger.info("[CDSE] downloaded %s -> %s (%.1f MB)",
                    product_id, out_path,
                    out_path.stat().st_size / 1024 / 1024)
        return out_path


class CopernicusDataSpaceAdapter(BaseAdapter):
    """Streaming AOI catalog poller for Sentinel-1/2/3/5P scenes.

    Same shape as SentinelHubAdapter — emits one observation per new scene.
    """
    adapter_type = "copernicus_dataspace"

    def __init__(self, config: AdapterConfig, mqtt_client=None):
        super().__init__(config, mqtt_client=mqtt_client)
        extra = config.extra or {}
        self._collection: str = extra.get("collection", "SENTINEL-1")
        self._bbox: list[float] = list(extra.get("bbox", [-125, 24, -66, 49]))
        self._max_age_days: int = int(extra.get("max_age_days", 7))
        self._limit_per_poll: int = int(extra.get("limit_per_poll", 50))
        self._max_cloud_cover = extra.get("max_cloud_cover")
        u = _resolve_env("CDSE_USERNAME")
        p = _resolve_env("CDSE_PASSWORD")
        self._tokens = _CDSETokenManager(u, p)
        self._seen: set[str] = set()

    async def connect(self) -> None:
        try:
            await self._tokens.get_token()
        except Exception as e:
            logger.error("[CDSE] %s auth failed: %s", self.config.adapter_id, e)
            raise

    async def disconnect(self) -> None:
        return

    async def stream_observations(self) -> AsyncIterator[dict]:
        while True:
            try:
                async for obs in self._poll():
                    yield obs
            except Exception as e:
                logger.warning("[CDSE] %s poll error: %s",
                               self.config.adapter_id, e)
            await asyncio.sleep(self.config.poll_interval_seconds)

    async def _poll(self) -> AsyncIterator[dict]:
        token = await self._tokens.get_token()
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=self._max_age_days)
        col = _COLLECTION_ALIASES.get(self._collection.lower(), self._collection)
        body = {
            "bbox": self._bbox,
            "datetime": f"{start.isoformat()}/{end.isoformat()}",
            "collections": [col],
            "limit": self._limit_per_poll,
        }
        if self._max_cloud_cover is not None and "SENTINEL-2" in col.upper():
            body["filter-lang"] = "cql2-json"
            body["filter"] = {"op": "<=",
                              "args": [{"property": "eo:cloud_cover"},
                                       float(self._max_cloud_cover)]}

        def _post():
            return requests.post(f"{CDSE_STAC_BASE}/search", json=body,
                                 headers={"Authorization": f"Bearer {token}",
                                          "User-Agent": "Heli.OS/1.0"},
                                 timeout=60)
        r = await asyncio.to_thread(_post)
        r.raise_for_status()
        for f in r.json().get("features") or []:
            sid = f.get("id")
            if not sid or sid in self._seen:
                continue
            self._seen.add(sid)
            yield self._scene_to_obs(f, col)

    def _scene_to_obs(self, feat: dict, collection: str) -> dict:
        props = feat.get("properties") or {}
        geom = feat.get("geometry") or {}
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
            "collection":   collection,
            "scene_id":     feat.get("id"),
            "datetime":     props.get("datetime"),
            "lat": cy, "lon": cx,
            "geometry":     geom,
            "cloud_cover":  props.get("eo:cloud_cover"),
            "platform":     props.get("platform") or props.get("constellation"),
            "metadata":     {k: v for k, v in props.items() if k != "datetime"},
        }
