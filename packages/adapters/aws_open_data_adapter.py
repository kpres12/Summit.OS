"""
Heli.OS — AWS Open Data Anonymous Buckets Adapter
======================================================
NOAA, NASA, USGS, and several federal partners publish bulk Earth-
observation data to public S3 buckets that are accessible **anonymously
with no auth at all**. This adapter exposes a thin streaming interface
over those buckets so Heli.OS can ingest:

  s3://sentinel-s2-l2a-cogs/        Sentinel-2 L2A cloud-optimized GeoTIFFs
  s3://sentinel-s1-l1c/             Sentinel-1 GRD scenes
  s3://landsat-pds/                 Landsat 8/9 archive (legacy)
  s3://usgs-landsat/                Landsat 8/9 (modern)
  s3://noaa-goes16/                 GOES-East geostationary, 5 min cadence
  s3://noaa-goes17/                 GOES-West (decommissioned but archived)
  s3://noaa-goes18/                 GOES-West replacement
  s3://noaa-mrms-pds/               MRMS multi-radar multi-sensor mosaics
  s3://noaa-nexrad-level2/          NEXRAD radar — already wired separately
  s3://spacenet-dataset/            SpaceNet building/road datasets

All accessible with anonymous HTTP — *no AWS credentials required* — via
the virtual-host endpoint `https://<bucket>.s3.amazonaws.com/<key>`.
S3 ListObjectsV2 also works anonymously for these buckets.

Adapter modes (extra.mode):
  poll_prefix   — periodic ListObjectsV2 against bucket/prefix; emit one
                  observation per new key (default)
  ondemand      — register-only

Config:
  bucket             — S3 bucket name
  prefix             — key prefix to list (e.g. "noaa-goes16/ABI-L1b-RadF/2026/115/")
  max_keys_per_poll  — cap per poll
  region             — AWS region (cosmetic; we use s3.amazonaws.com endpoint)
  emit_url           — if true (default), include a public HTTPS URL so
                       downstream consumers can fetch without S3 SDK

Example registration:
  {
    "adapter_type": "aws_open_data",
    "name": "GOES-16 ABI-L1b RadF (full-disk, last poll)",
    "poll_interval_seconds": 600,
    "extra": {
      "bucket": "noaa-goes16",
      "prefix": "ABI-L1b-RadF/2026/",
      "max_keys_per_poll": 500
    }
  }

For one-off catalog/index queries (training loaders), use the
AWSOpenDataClient class directly.
"""

from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Optional
from urllib.parse import quote

try:
    import requests
except ImportError as e:
    raise ImportError("aws_open_data_adapter requires `requests`") from e

from .base import BaseAdapter, AdapterConfig

logger = logging.getLogger("heli.adapters.aws_open_data")


KNOWN_BUCKETS = {
    "sentinel-s2-l2a-cogs":   "Sentinel-2 L2A cloud-optimized GeoTIFFs",
    "sentinel-s1-l1c":        "Sentinel-1 GRD",
    "landsat-pds":            "Landsat 8/9 (legacy PDS)",
    "usgs-landsat":           "Landsat 8/9 (modern)",
    "noaa-goes16":            "GOES-East geostationary (5-min)",
    "noaa-goes17":            "GOES-West (decommissioned, archived)",
    "noaa-goes18":            "GOES-West (current)",
    "noaa-mrms-pds":          "MRMS multi-radar multi-sensor",
    "noaa-nexrad-level2":     "NEXRAD radar Level-II",
    "spacenet-dataset":       "SpaceNet building/road datasets",
}


def _bucket_endpoint(bucket: str) -> str:
    return f"https://{bucket}.s3.amazonaws.com"


class AWSOpenDataClient:
    """Synchronous anonymous-S3 client. Uses ListObjectsV2 + virtual-host GETs."""

    @staticmethod
    def list_objects(bucket: str, prefix: str, max_keys: int = 1000,
                     continuation_token: Optional[str] = None
                     ) -> tuple[list[dict], Optional[str]]:
        """List objects under a prefix anonymously.

        Returns (rows, next_continuation_token). Each row is
        {key, size, last_modified}.
        """
        url = (f"{_bucket_endpoint(bucket)}/?prefix={quote(prefix)}"
               f"&list-type=2&max-keys={max_keys}")
        if continuation_token:
            url += f"&continuation-token={quote(continuation_token)}"
        r = requests.get(url, timeout=60,
                         headers={"User-Agent": "Heli.OS/1.0"})
        r.raise_for_status()
        ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
        root = ET.fromstring(r.text)

        rows: list[dict] = []
        for c in root.findall(f"{ns}Contents"):
            key_el  = c.find(f"{ns}Key")
            size_el = c.find(f"{ns}Size")
            lm_el   = c.find(f"{ns}LastModified")
            if key_el is None or not key_el.text:
                continue
            rows.append({
                "key":           key_el.text,
                "size":          int(size_el.text) if size_el is not None and size_el.text else 0,
                "last_modified": lm_el.text if lm_el is not None else None,
            })

        next_tok: Optional[str] = None
        truncated = root.find(f"{ns}IsTruncated")
        if truncated is not None and truncated.text == "true":
            tok_el = root.find(f"{ns}NextContinuationToken")
            if tok_el is not None and tok_el.text:
                next_tok = tok_el.text

        return rows, next_tok

    @staticmethod
    def list_all(bucket: str, prefix: str, hard_limit: int = 50_000) -> list[dict]:
        """List up to hard_limit objects across paged ListObjectsV2 calls."""
        all_rows: list[dict] = []
        token: Optional[str] = None
        while True:
            rows, token = AWSOpenDataClient.list_objects(
                bucket, prefix, max_keys=1000, continuation_token=token)
            all_rows.extend(rows)
            if not token or len(all_rows) >= hard_limit:
                break
        return all_rows[:hard_limit]

    @staticmethod
    def public_url(bucket: str, key: str) -> str:
        return f"{_bucket_endpoint(bucket)}/{quote(key)}"

    @staticmethod
    def download(bucket: str, key: str, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        url = AWSOpenDataClient.public_url(bucket, key)
        with requests.get(url, stream=True, timeout=300,
                          headers={"User-Agent": "Heli.OS/1.0"}) as r:
            r.raise_for_status()
            with out_path.open("wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
        return out_path


class AWSOpenDataAdapter(BaseAdapter):
    adapter_type = "aws_open_data"

    def __init__(self, config: AdapterConfig, mqtt_client=None):
        super().__init__(config, mqtt_client=mqtt_client)
        extra = config.extra or {}
        self._bucket: str = extra.get("bucket", "noaa-goes16")
        self._prefix: str = extra.get("prefix", "")
        self._max: int = int(extra.get("max_keys_per_poll", 500))
        self._emit_url: bool = bool(extra.get("emit_url", True))
        self._mode: str = extra.get("mode", "poll_prefix")
        self._seen: set[str] = set()

    async def connect(self) -> None:
        return

    async def disconnect(self) -> None:
        return

    async def stream_observations(self) -> AsyncIterator[dict]:
        if self._mode != "poll_prefix":
            while True:
                await asyncio.sleep(60)
            return
        while True:
            try:
                async for obs in self._poll():
                    yield obs
            except Exception as e:
                logger.warning("[aws_open_data] %s poll error: %s",
                               self.config.adapter_id, e)
            await asyncio.sleep(self.config.poll_interval_seconds)

    async def _poll(self) -> AsyncIterator[dict]:
        rows, _ = await asyncio.to_thread(
            AWSOpenDataClient.list_objects, self._bucket, self._prefix, self._max)
        n_new = 0
        for r in rows:
            k = r["key"]
            if k in self._seen:
                continue
            self._seen.add(k)
            n_new += 1
            obs = {
                "adapter_type":  self.adapter_type,
                "adapter_id":    self.config.adapter_id,
                "ts":            datetime.now(timezone.utc).isoformat(),
                "entity_type":   "data_object",
                "asset_type":    "s3_object",
                "bucket":        self._bucket,
                "key":           k,
                "size":          r.get("size"),
                "last_modified": r.get("last_modified"),
                "description":   KNOWN_BUCKETS.get(self._bucket),
            }
            if self._emit_url:
                obs["url"] = AWSOpenDataClient.public_url(self._bucket, k)
            yield obs
        if len(self._seen) > 100_000:
            self._seen = set(list(self._seen)[-50_000:])
        logger.info("[aws_open_data] %s -> %d new keys",
                    self.config.adapter_id, n_new)
