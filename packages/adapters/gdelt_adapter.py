"""
Heli.OS — GDELT 2.0 Conflict & Events Adapter
=================================================
GDELT (Global Database of Events, Language, and Tone) is a fully open,
no-auth feed of geocoded global events extracted from worldwide news
media. Updated every 15 minutes with the previous quarter-hour's events.

Free, no key, programmatic. Direct HTTP CSV download.

Useful for:
  - Global situational awareness briefings (COMMAND view feed)
  - Conflict / unrest / disaster context overlay on operational map
  - Pattern-of-life enrichment for federal customers
  - Civilian crisis response (earthquakes, floods, civil unrest)

Endpoints used:
  Index of latest 15-min files:
    http://data.gdeltproject.org/gdeltv2/lastupdate.txt
  Each file is a CSV gzipped, listed as a URL on the line.
  Schema docs: http://data.gdeltproject.org/documentation/GDELT-Event_Codebook-V2.0.pdf

Each event row is a CAMEO-coded interaction between two actors with:
  - GLOBALEVENTID, SQLDATE, ActionGeo_Lat / Lon, ActionGeo_FullName,
  - EventCode (CAMEO 4-digit), QuadClass (1-4: Verbal/Material × Coop/Conflict),
  - GoldsteinScale (-10 to +10, conflict↔cooperation),
  - NumMentions, AvgTone, SourceURL, ...

Adapter modes (AdapterConfig.extra.mode):
  poll_recent — poll lastupdate.txt every poll_interval_seconds, ingest
                each new event as an observation.

Config (AdapterConfig.extra):
  bbox        — optional [lon_min, lat_min, lon_max, lat_max] filter
  min_goldstein_negative — only ingest events with Goldstein score
                           below this (default: -5; conflict-only filter)
  event_codes — optional list of CAMEO codes to allow

Register example:
  {
    "adapter_type": "gdelt",
    "name": "GDELT — Global Conflict",
    "poll_interval_seconds": 900,    # 15 min
    "extra": {
      "min_goldstein_negative": -5,
      "bbox": null
    }
  }
"""

from __future__ import annotations

import asyncio
import csv
import gzip
import io
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

try:
    import requests
except ImportError as e:
    raise ImportError("gdelt_adapter requires `requests`") from e

from .base import BaseAdapter, AdapterConfig

logger = logging.getLogger("heli.adapters.gdelt")


GDELT_LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
GDELT_EVENT_HEADERS = [
    "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources",
    "NumArticles", "AvgTone",
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code", "Actor1Geo_Lat",
    "Actor1Geo_Long", "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code", "Actor2Geo_Lat",
    "Actor2Geo_Long", "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_ADM2Code", "ActionGeo_Lat",
    "ActionGeo_Long", "ActionGeo_FeatureID",
    "DATEADDED", "SOURCEURL",
]

QUAD_CLASS_LABELS = {
    1: "verbal_cooperation",
    2: "material_cooperation",
    3: "verbal_conflict",
    4: "material_conflict",
}


def _parse_lastupdate(text: str) -> Optional[str]:
    """Find the gdeltv2/*.export.CSV.zip / .CSV.gz line in lastupdate.txt."""
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            url = parts[-1]
            if "export.CSV" in url:
                return url
    return None


def _bbox_contains(bbox: list[float], lat: float, lon: float) -> bool:
    return bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]


class GDELTAdapter(BaseAdapter):
    adapter_type = "gdelt"

    def __init__(self, config: AdapterConfig, mqtt_client=None):
        super().__init__(config, mqtt_client=mqtt_client)
        extra = config.extra or {}
        self._bbox: Optional[list[float]] = extra.get("bbox")
        self._min_goldstein_negative: float = float(
            extra.get("min_goldstein_negative", -5.0))
        self._event_codes: Optional[set[str]] = (
            set(extra["event_codes"]) if extra.get("event_codes") else None)
        self._seen_files: set[str] = set()
        self._seen_event_ids: set[str] = set()
        self._max_seen_size: int = 100_000  # bound the seen-set

    async def connect(self) -> None:
        return

    async def disconnect(self) -> None:
        return

    async def stream_observations(self) -> AsyncIterator[dict]:
        while True:
            try:
                async for obs in self._poll_latest():
                    yield obs
            except Exception as e:
                logger.warning("[GDELT] %s poll error: %s",
                               self.config.adapter_id, e)
            await asyncio.sleep(self.config.poll_interval_seconds)

    async def _poll_latest(self) -> AsyncIterator[dict]:
        def _fetch(url: str, decode: bool = True):
            r = requests.get(url, timeout=60,
                             headers={"User-Agent": "Heli.OS/1.0"})
            r.raise_for_status()
            return r.text if decode else r.content

        idx_text = await asyncio.to_thread(_fetch, GDELT_LASTUPDATE_URL)
        url = _parse_lastupdate(idx_text)
        if not url:
            logger.warning("[GDELT] lastupdate index had no export.CSV URL")
            return
        if url in self._seen_files:
            return
        self._seen_files.add(url)
        if len(self._seen_files) > 1000:
            self._seen_files = set(list(self._seen_files)[-500:])

        raw = await asyncio.to_thread(_fetch, url, False)
        # File can be either .zip or .gz depending on flavor; handle both.
        text = ""
        if url.endswith(".gz"):
            try:
                text = gzip.decompress(raw).decode("latin-1", errors="replace")
            except Exception:
                pass
        if not text and url.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                names = [n for n in zf.namelist() if n.endswith(".CSV")]
                if names:
                    text = zf.read(names[0]).decode("latin-1", errors="replace")
        if not text:
            logger.warning("[GDELT] could not decode %s", url)
            return

        reader = csv.reader(io.StringIO(text), delimiter="\t")
        n_emitted = 0
        for row in reader:
            if len(row) < len(GDELT_EVENT_HEADERS):
                continue
            rec = dict(zip(GDELT_EVENT_HEADERS, row))
            ev_id = rec.get("GLOBALEVENTID", "")
            if not ev_id or ev_id in self._seen_event_ids:
                continue

            try:
                lat = float(rec.get("ActionGeo_Lat") or 0)
                lon = float(rec.get("ActionGeo_Long") or 0)
            except ValueError:
                continue
            if lat == 0 and lon == 0:
                continue
            if self._bbox and not _bbox_contains(self._bbox, lat, lon):
                continue

            try:
                gs = float(rec.get("GoldsteinScale") or 0)
            except ValueError:
                gs = 0.0
            if gs > self._min_goldstein_negative:
                # Skip non-conflict events (Goldstein > threshold means cooperation)
                continue

            event_code = rec.get("EventCode") or ""
            if self._event_codes and event_code not in self._event_codes:
                continue

            try:
                quad = int(rec.get("QuadClass") or 0)
            except ValueError:
                quad = 0

            self._seen_event_ids.add(ev_id)
            if len(self._seen_event_ids) > self._max_seen_size:
                self._seen_event_ids = set(list(self._seen_event_ids)[-self._max_seen_size // 2:])

            yield {
                "adapter_type":  self.adapter_type,
                "adapter_id":    self.config.adapter_id,
                "ts":            datetime.now(timezone.utc).isoformat(),
                "entity_type":   "event_observation",
                "asset_type":    "geopolitical_event",
                "event_id":      ev_id,
                "lat":           lat,
                "lon":           lon,
                "place":         rec.get("ActionGeo_FullName"),
                "country":       rec.get("ActionGeo_CountryCode"),
                "event_code":    event_code,
                "event_root":    rec.get("EventRootCode"),
                "quad_class":    QUAD_CLASS_LABELS.get(quad, "unknown"),
                "goldstein":     gs,
                "actor1":        rec.get("Actor1Name"),
                "actor1_country": rec.get("Actor1CountryCode"),
                "actor2":        rec.get("Actor2Name"),
                "actor2_country": rec.get("Actor2CountryCode"),
                "n_mentions":    int(rec.get("NumMentions") or 0),
                "avg_tone":      float(rec.get("AvgTone") or 0),
                "source_url":    rec.get("SOURCEURL"),
                "date":          rec.get("SQLDATE"),
            }
            n_emitted += 1

        logger.info("[GDELT] %s ingested %d events from %s",
                    self.config.adapter_id, n_emitted, url.rsplit("/", 1)[-1])
