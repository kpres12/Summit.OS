"""
packages/utm/notam.py — FAA NOTAM API client.

Fetches active NOTAMs (Notices to Air Missions) for a given area and converts
airspace-restriction NOTAMs into OPA-compatible exclusion geofence objects.

API:  https://api.faa.gov/notamSearch/notams
Auth: Free API key from https://developer.faa.gov (set FAA_API_KEY env var).
      Without a key, the client returns an empty list gracefully — missions can
      still be created but will not have NOTAM awareness.

NOTAM types handled:
  D  — Domestic (flight restrictions, TFRs, airspace closures)
  FDC— Flight Data Center (instrument procedure changes, airspace amendments)

Converted to OPA exclusion zones:
  - TFRs (Temporary Flight Restrictions) — NOTAM keyword "TFR"
  - Airspace closures — keyword "airspace closed" / "prohibited"
  - Presidential/VIP NOTAMs — keyword "POTUS" / "VVIP"
  - Wildfire TFRs — keyword "fire" + airspace restriction

Reference: https://www.faa.gov/air_traffic/publications/atpubs/notam_html/
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("utm.notam")

_FAA_NOTAM_URL = "https://api.faa.gov/notamSearch/notams"
_DEFAULT_RADIUS_NM = 25
_REQUEST_TIMEOUT = 10

# NOTAM text patterns that indicate an airspace restriction worth enforcing
_RESTRICTION_KEYWORDS = (
    "tfr",
    "temporary flight restriction",
    "prohibited",
    "airspace closed",
    "no fly",
    "flight restriction",
    "potus",
    "vvip",
    "fire",
    "disaster",
    "national security",
    "special security",
    "stadium",
    "nuclear",
    "toxic",
)


@dataclass
class Notam:
    notam_id: str
    classification: str          # D, FDC, INTERNATIONAL, etc.
    effective_start: str         # ISO timestamp
    effective_end: Optional[str]
    location: str                # ICAO facility identifier
    text: str                    # Raw NOTAM text
    lat: Optional[float]
    lon: Optional[float]
    radius_nm: Optional[float]
    is_restriction: bool
    tfr_type: Optional[str]      # "POTUS", "FIRE", "SECURITY", "STADIUM", etc.

    def to_exclusion_geofence(self, geofence_id_prefix: str = "notam") -> Optional[Dict[str, Any]]:
        """
        Convert this NOTAM to an OPA-compatible exclusion geofence bounding box.

        Returns None if the NOTAM lacks coordinate data.
        The geofence uses a square bbox approximation around the NOTAM center.
        For circular NOTAMs (lat/lon + radius), the bbox is the bounding square.
        """
        if self.lat is None or self.lon is None:
            return None

        radius_nm = self.radius_nm or 1.0
        # 1 nautical mile ≈ 0.01667 degrees latitude; longitude varies by lat
        import math
        deg_lat = radius_nm * (1.0 / 60.0)
        deg_lon = radius_nm * (1.0 / 60.0) / max(math.cos(math.radians(self.lat)), 0.01)

        return {
            "geofence_id": f"{geofence_id_prefix}-{self.notam_id}",
            "type": "exclusion",
            "source": "notam",
            "notam_id": self.notam_id,
            "description": self.text[:120],
            "effective_start": self.effective_start,
            "effective_end": self.effective_end,
            "min_lat": self.lat - deg_lat,
            "max_lat": self.lat + deg_lat,
            "min_lon": self.lon - deg_lon,
            "max_lon": self.lon + deg_lon,
        }


def _classify_tfr(text: str) -> Optional[str]:
    t = text.lower()
    if "potus" in t or "president" in t:
        return "POTUS"
    if "fire" in t or "wildfire" in t:
        return "FIRE"
    if "stadium" in t or "sporting" in t:
        return "STADIUM"
    if "security" in t or "national security" in t:
        return "SECURITY"
    if "nuclear" in t or "radiolog" in t:
        return "HAZMAT"
    if "disaster" in t or "emergency" in t:
        return "DISASTER"
    return "GENERAL"


def _is_restriction(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in _RESTRICTION_KEYWORDS)


def _parse_notam(raw: Dict[str, Any]) -> Notam:
    """Parse a single NOTAM object from the FAA API response."""
    props = raw.get("properties", raw)
    geo = raw.get("geometry") or {}
    coords = geo.get("coordinates") or []

    lat: Optional[float] = None
    lon: Optional[float] = None
    if coords and len(coords) >= 2:
        try:
            lon, lat = float(coords[0]), float(coords[1])
        except (TypeError, ValueError):
            pass

    text = (
        props.get("coreNOTAMData", {}).get("notam", {}).get("text", "")
        or props.get("text", "")
        or props.get("notamText", "")
        or ""
    )
    notam_id = (
        props.get("coreNOTAMData", {}).get("notam", {}).get("id", "")
        or props.get("id", "")
        or raw.get("id", "unknown")
    )
    classification = (
        props.get("coreNOTAMData", {}).get("notam", {}).get("classification", "")
        or props.get("classification", "D")
    )
    effective_start = (
        props.get("coreNOTAMData", {}).get("notam", {}).get("effectiveStart", "")
        or props.get("effectiveStart", "")
        or datetime.now(timezone.utc).isoformat()
    )
    effective_end = (
        props.get("coreNOTAMData", {}).get("notam", {}).get("effectiveEnd")
        or props.get("effectiveEnd")
    )
    location = (
        props.get("coreNOTAMData", {}).get("notam", {}).get("location", "")
        or props.get("location", "")
        or "UNKN"
    )

    # Radius from NOTAM radius field or geometry
    radius_nm: Optional[float] = None
    try:
        r = (
            props.get("coreNOTAMData", {}).get("notam", {}).get("radius")
            or props.get("radius")
        )
        if r is not None:
            radius_nm = float(r)
    except (TypeError, ValueError):
        pass

    is_restr = _is_restriction(text)
    tfr_type = _classify_tfr(text) if is_restr else None

    return Notam(
        notam_id=str(notam_id),
        classification=classification,
        effective_start=effective_start,
        effective_end=effective_end,
        location=location,
        text=text,
        lat=lat,
        lon=lon,
        radius_nm=radius_nm,
        is_restriction=is_restr,
        tfr_type=tfr_type,
    )


class NotamClient:
    """
    Async-compatible FAA NOTAM client.

    Uses stdlib urllib (no external HTTP deps).
    Caches results in memory for `cache_ttl_seconds` to avoid hammering the API
    on back-to-back mission creations in the same area.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl_seconds: int = 300,
        persistent_cache=None,
    ) -> None:
        self._api_key = api_key or os.getenv("FAA_API_KEY", "")
        self._mem_cache: Dict[str, tuple[float, List[Notam]]] = {}
        self._cache_ttl = cache_ttl_seconds
        # Persistent SQLite cache — survives restarts, works offline
        self._disk: Optional[Any] = persistent_cache
        if self._disk is None:
            try:
                from .cache import get_cache
                self._disk = get_cache()
            except Exception:
                self._disk = None

    def _cache_key(self, lat: float, lon: float, radius_nm: float) -> str:
        return f"{lat:.3f},{lon:.3f},{radius_nm:.1f}"

    def _mem_get(self, key: str) -> Optional[List[Notam]]:
        import time
        entry = self._mem_cache.get(key)
        if entry and (time.time() - entry[0]) < self._cache_ttl:
            return entry[1]
        return None

    def _mem_set(self, key: str, notams: List[Notam]) -> None:
        import time
        self._mem_cache[key] = (time.time(), notams)

    def fetch(
        self,
        lat: float,
        lon: float,
        radius_nm: float = _DEFAULT_RADIUS_NM,
        notam_type: str = "D",
    ) -> List[Notam]:
        """
        Fetch active NOTAMs within radius_nm nautical miles of (lat, lon).

        Cache hierarchy:
          1. In-memory (5 min TTL) — fastest, avoids repeated API calls
          2. SQLite disk cache (4 hr stale TTL) — survives restarts, works offline
          3. FAA API — live data when internet is available
          4. Empty list — fail-open if all sources unavailable
        """
        key = self._cache_key(lat, lon, radius_nm)

        # 1. Memory cache (hot path)
        mem = self._mem_get(key)
        if mem is not None:
            return mem

        # 2. Disk cache — use if fresh; fall through to API refresh if stale
        disk_notams: Optional[List[Notam]] = None
        disk_is_fresh = False
        if self._disk is not None:
            try:
                raw_list, disk_is_fresh = self._disk.get_notams(key)
                if raw_list is not None:
                    disk_notams = [_parse_notam(r) for r in raw_list]
                    if disk_is_fresh:
                        self._mem_set(key, disk_notams)
                        return disk_notams
                    # Stale — try API but fall back to disk if API fails
            except Exception as exc:
                logger.debug("NOTAM disk cache read error: %s", exc)

        # 3. Live API fetch
        if not self._api_key:
            if disk_notams is not None:
                logger.warning(
                    "FAA_API_KEY not set — serving stale NOTAM cache for %.4f,%.4f",
                    lat, lon,
                )
                self._mem_set(key, disk_notams)
                return disk_notams
            logger.warning(
                "FAA_API_KEY not set — NOTAM awareness disabled. "
                "Register free at developer.faa.gov"
            )
            return []

        params = urllib.parse.urlencode({
            "lat": f"{lat:.5f}",
            "long": f"{lon:.5f}",
            "radius": f"{radius_nm:.1f}",
            "notamType": notam_type,
            "sortBy": "effectiveStartDate",
            "sortOrder": "Desc",
            "pageSize": "100",
        })
        url = f"{_FAA_NOTAM_URL}?{params}"
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "client_id": self._api_key,
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            logger.warning("FAA NOTAM API error %s", exc.code)
            if disk_notams is not None:
                logger.warning("Serving stale NOTAM cache (age unknown) for %.4f,%.4f", lat, lon)
                self._mem_set(key, disk_notams)
                return disk_notams
            return []
        except Exception as exc:
            logger.warning("FAA NOTAM API unreachable: %s", exc)
            if disk_notams is not None:
                logger.warning(
                    "OFFLINE MODE: serving cached NOTAMs for %.4f,%.4f "
                    "(may be stale — verify before flight)",
                    lat, lon,
                )
                self._mem_set(key, disk_notams)
                return disk_notams
            return []

        items = data.get("items", data) if isinstance(data, dict) else data
        if not isinstance(items, list):
            items = []

        notams: List[Notam] = []
        raw_for_cache = []
        for raw in items:
            try:
                notams.append(_parse_notam(raw))
                raw_for_cache.append(raw)
            except Exception as exc:
                logger.debug("Failed to parse NOTAM: %s", exc)

        # Write to both caches
        self._mem_set(key, notams)
        if self._disk is not None:
            try:
                self._disk.set_notams(key, raw_for_cache)
            except Exception as exc:
                logger.debug("NOTAM disk cache write error: %s", exc)

        logger.info(
            "Fetched %d NOTAMs (%.1fnm radius around %.4f,%.4f), %d are restrictions",
            len(notams), radius_nm, lat, lon,
            sum(1 for n in notams if n.is_restriction),
        )
        return notams

    def restriction_geofences(
        self,
        lat: float,
        lon: float,
        radius_nm: float = _DEFAULT_RADIUS_NM,
    ) -> List[Dict[str, Any]]:
        """
        Return OPA-compatible exclusion geofence dicts for all active
        airspace-restriction NOTAMs in the area. Ready to inject into
        OPA evaluate_geofence() context.
        """
        notams = self.fetch(lat, lon, radius_nm)
        geofences = []
        for n in notams:
            if n.is_restriction:
                gf = n.to_exclusion_geofence()
                if gf:
                    geofences.append(gf)
        return geofences
