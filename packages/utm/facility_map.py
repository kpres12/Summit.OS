"""
packages/utm/facility_map.py — FAA UAS Facility Map client.

The FAA UAS Facility Map defines, for each 1-arc-minute grid cell in the US,
the maximum altitude (in feet AGL) at which Part 107 drone operations are
authorized WITHOUT prior FAA approval (LAANC authorization).

  0 ft  → LAANC authorization required before any flight
  >0 ft → Can fly up to that altitude under Part 107 without approval

Data source: FAA ArcGIS REST API (public, no auth required)
  https://faa.maps.arcgis.com/sharing/rest/content/items/...

The client downloads the facility map grid for a bounding box, caches it
locally (default: /tmp/summit_facility_map_cache/), and returns:
  - The authorized altitude at a specific lat/lon
  - An OPA-compatible altitude-restriction geofence if altitude is 0

Key regulations encoded:
  FAA Part 107.51 — Operations over moving vehicles
  FAA Part 107.51(b)(1) — Max altitude 400ft AGL (122m)
  14 CFR Part 77 — Obstruction standards around airports

Reference: https://www.faa.gov/uas/programs_partnerships/data_exchange
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("utm.facility_map")

# FAA ArcGIS Feature Service — UAS Facility Maps (public layer)
_FACILITY_MAP_URL = (
    "https://services6.arcgis.com/ssFJjBXIUyZDrSYZ/arcgis/rest/services/"
    "UAS_Facility_Map_Data_Public_View/FeatureServer/0/query"
)
_CACHE_DIR = os.getenv("SUMMIT_UTM_CACHE_DIR", "/tmp/summit_facility_map_cache")
_CACHE_TTL_SECONDS = 86400  # 24 hours — facility maps change infrequently
_REQUEST_TIMEOUT = 15

# Default Part 107 limits when facility map is unavailable
_DEFAULT_AUTH_ALTITUDE_FT = 400  # FAA Part 107 max
_CLASS_B_ALTITUDE_FT = 0         # Class B airspace — LAANC required


@dataclass
class FacilityCell:
    """A single UAS Facility Map grid cell."""
    cell_id: str
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    authorized_altitude_ft: int   # 0 = LAANC required; >0 = authorized up to this altitude
    airport_id: Optional[str]     # ICAO identifier of the controlling airport
    airspace_class: Optional[str] # B, C, D, E, G

    @property
    def laanc_required(self) -> bool:
        return self.authorized_altitude_ft == 0

    def to_altitude_restriction_geofence(self) -> Optional[Dict[str, Any]]:
        """
        Returns an OPA exclusion geofence if this cell requires LAANC (altitude = 0).
        For cells with a reduced altitude, returns metadata but not an exclusion zone
        (the altitude limit is enforced by the geofence.rego max_altitude_m check).
        """
        if not self.laanc_required:
            return None
        return {
            "geofence_id": f"facility-{self.cell_id}",
            "type": "exclusion",
            "source": "faa_facility_map",
            "description": (
                f"LAANC authorization required — "
                f"{self.airspace_class or 'controlled'} airspace"
                + (f" near {self.airport_id}" if self.airport_id else "")
            ),
            "authorized_altitude_ft": self.authorized_altitude_ft,
            "airport_id": self.airport_id,
            "airspace_class": self.airspace_class,
            "min_lat": self.min_lat,
            "max_lat": self.max_lat,
            "min_lon": self.min_lon,
            "max_lon": self.max_lon,
        }


def _bbox_from_center(lat: float, lon: float, radius_nm: float) -> Tuple[float, float, float, float]:
    """Return (min_lon, min_lat, max_lon, max_lat) bounding box."""
    deg_lat = radius_nm / 60.0
    deg_lon = radius_nm / 60.0 / max(math.cos(math.radians(lat)), 0.01)
    return (lon - deg_lon, lat - deg_lat, lon + deg_lon, lat + deg_lat)


def _cache_path(lat: float, lon: float, radius_nm: float) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"{lat:.2f}_{lon:.2f}_{radius_nm:.1f}.json")


def _load_cache(path: str) -> Optional[List[Dict]]:
    try:
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            if time.time() - mtime < _CACHE_TTL_SECONDS:
                with open(path) as f:
                    return json.load(f)
    except Exception:
        pass
    return None


def _save_cache(path: str, data: List[Dict]) -> None:
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


class FacilityMapClient:
    """
    FAA UAS Facility Map client.

    Queries the public FAA ArcGIS feature service for UAS facility map cells
    intersecting a given bounding box. Results are cached on disk for 24 hours.
    """

    def fetch_cells(
        self,
        lat: float,
        lon: float,
        radius_nm: float = 5.0,
    ) -> List[FacilityCell]:
        """
        Fetch facility map cells within radius_nm nautical miles of (lat, lon).
        Returns empty list on error — fail-open for availability.
        """
        cache_path = _cache_path(lat, lon, radius_nm)
        cached = _load_cache(cache_path)
        if cached is not None:
            return [self._parse_cell(c) for c in cached]

        min_lon, min_lat, max_lon, max_lat = _bbox_from_center(lat, lon, radius_nm)

        params = urllib.request.urlencode if hasattr(urllib.request, "urlencode") else None
        import urllib.parse as _up
        query = _up.urlencode({
            "where": "1=1",
            "geometry": f"{min_lon},{min_lat},{max_lon},{max_lat}",
            "geometryType": "esriGeometryEnvelope",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "IDENT,CEILING,AIRSPACE_CLASS,LOWER_LIMIT,UPPER_LIMIT",
            "f": "json",
            "returnGeometry": "true",
        })
        url = f"{_FACILITY_MAP_URL}?{query}"

        try:
            with urllib.request.urlopen(url, timeout=_REQUEST_TIMEOUT) as resp:
                data = json.loads(resp.read())
        except Exception as exc:
            logger.warning(
                "FAA Facility Map API unreachable: %s — using default 400ft limit", exc
            )
            return []

        features = data.get("features", [])
        raw_cells = []
        cells: List[FacilityCell] = []

        for feat in features:
            try:
                raw_cells.append(feat)
                cells.append(self._parse_feature(feat))
            except Exception as exc:
                logger.debug("Failed to parse facility cell: %s", exc)

        _save_cache(cache_path, raw_cells)
        logger.info(
            "Fetched %d facility map cells (%.1fnm radius around %.4f,%.4f), "
            "%d require LAANC",
            len(cells), radius_nm, lat, lon,
            sum(1 for c in cells if c.laanc_required),
        )
        return cells

    def _parse_feature(self, feat: Dict) -> FacilityCell:
        attrs = feat.get("attributes", {})
        geom = feat.get("geometry", {})
        rings = geom.get("rings", [[]])[0]

        lats = [p[1] for p in rings if len(p) >= 2]
        lons = [p[0] for p in rings if len(p) >= 2]

        ceiling_raw = attrs.get("CEILING") or attrs.get("UPPER_LIMIT") or 400
        try:
            ceiling = int(ceiling_raw)
        except (TypeError, ValueError):
            ceiling = 400

        return FacilityCell(
            cell_id=str(attrs.get("OBJECTID", attrs.get("IDENT", "unknown"))),
            min_lat=min(lats) if lats else 0.0,
            max_lat=max(lats) if lats else 0.0,
            min_lon=min(lons) if lons else 0.0,
            max_lon=max(lons) if lons else 0.0,
            authorized_altitude_ft=ceiling,
            airport_id=attrs.get("IDENT"),
            airspace_class=attrs.get("AIRSPACE_CLASS"),
        )

    def _parse_cell(self, raw: Dict) -> FacilityCell:
        """Re-parse a cached raw feature dict."""
        return self._parse_feature(raw)

    def authorized_altitude_ft(self, lat: float, lon: float) -> int:
        """
        Return the Part 107 authorized altitude (ft AGL) at the given point.
        Returns 400 (default FAA limit) if no facility map data is available.
        Returns 0 if the point is in a cell requiring LAANC authorization.
        """
        cells = self.fetch_cells(lat, lon, radius_nm=1.0)
        for cell in cells:
            if (cell.min_lat <= lat <= cell.max_lat and
                    cell.min_lon <= lon <= cell.max_lon):
                return cell.authorized_altitude_ft
        return _DEFAULT_AUTH_ALTITUDE_FT

    def exclusion_geofences(
        self, lat: float, lon: float, radius_nm: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        Return OPA-compatible exclusion geofences for cells requiring LAANC.
        """
        cells = self.fetch_cells(lat, lon, radius_nm)
        result = []
        for cell in cells:
            gf = cell.to_altitude_restriction_geofence()
            if gf:
                result.append(gf)
        return result
