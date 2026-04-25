"""
Digital Elevation Model (DEM) Provider for Heli.OS

Serves elevation lookups from SRTM .hgt tile files.
SRTM1 (1 arc-second ≈ 30m resolution) and SRTM3 (3 arc-sec ≈ 90m) supported.

Tile naming convention: N34W118.hgt
SRTM1 tiles:  3601 × 3601 int16 samples (big-endian)
SRTM3 tiles:  1201 × 1201 int16 samples (big-endian)

Tiles are freely available from:
  - NASA Earthdata: https://dwtkns.com/srtm30m/
  - USGS EarthExplorer
  - pip install elevation && eio clip -o dem.tif --bounds ...

Usage:
    dem = DEMProvider("/data/dem")
    elev = dem.get_elevation(34.052, -118.243)   # metres MSL
    los  = dem.check_line_of_sight(34.0, -118.0, 100, 34.1, -118.1, 10)
    profile = dem.get_elevation_profile(34.0, -118.0, 34.1, -118.1, n_samples=50)
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("geo.dem")

try:
    import numpy as np

    _NP = True
except ImportError:
    np = None  # type: ignore
    _NP = False


class DEMProvider:
    """
    Elevation lookups from SRTM .hgt tiles.

    Tiles are loaded on first access and cached in memory.
    Returns 0.0 gracefully when no tile is available (fail-open).
    """

    def __init__(self, dem_dir: str = "/data/dem"):
        self._dem_dir = dem_dir
        self._cache: Dict[str, Optional[object]] = {}  # tile_key → numpy array or None

    # ── Public API ────────────────────────────────────────────────────────────

    def get_elevation(self, lat: float, lon: float) -> float:
        """
        Return ground elevation in metres MSL at (lat, lon).
        Returns 0.0 if no tile is available or numpy is not installed.
        """
        if not _NP:
            return 0.0
        tile = self._load_tile(lat, lon)
        if tile is None:
            return 0.0
        return float(self._sample(tile, lat, lon))

    def get_elevation_profile(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        n_samples: int = 50,
    ) -> List[Tuple[float, float]]:
        """
        Elevation profile along a great-circle path.

        Returns list of (distance_m, elevation_m) from the start point.
        """
        results = []
        for i in range(n_samples + 1):
            t = i / n_samples
            lat = lat1 + t * (lat2 - lat1)
            lon = lon1 + t * (lon2 - lon1)
            elev = self.get_elevation(lat, lon)
            dist = _haversine_m(lat1, lon1, lat, lon)
            results.append((dist, elev))
        return results

    def check_line_of_sight(
        self,
        obs_lat: float,
        obs_lon: float,
        obs_alt_m: float,
        tgt_lat: float,
        tgt_lon: float,
        tgt_alt_m: float,
        n_samples: int = 50,
    ) -> bool:
        """
        Return True if the line of sight between observer and target is
        unobstructed by terrain. Uses a simple ray-terrain intersection test.
        """
        total_dist = _haversine_m(obs_lat, obs_lon, tgt_lat, tgt_lon)
        if total_dist < 1.0:
            return True

        obs_ground = self.get_elevation(obs_lat, obs_lon)
        tgt_ground = self.get_elevation(tgt_lat, tgt_lon)
        obs_abs = obs_ground + obs_alt_m
        tgt_abs = tgt_ground + tgt_alt_m

        for i in range(1, n_samples):
            t = i / n_samples
            lat = obs_lat + t * (tgt_lat - obs_lat)
            lon = obs_lon + t * (tgt_lon - obs_lon)
            terrain = self.get_elevation(lat, lon)
            ray_alt = obs_abs + t * (tgt_abs - obs_abs)
            if terrain > ray_alt:
                return False
        return True

    def terrain_following_altitude(
        self,
        lat: float,
        lon: float,
        agl_m: float = 30.0,
    ) -> float:
        """
        Return absolute altitude (MSL) required to maintain agl_m above terrain.
        """
        return self.get_elevation(lat, lon) + agl_m

    # ── Tile loading ─────────────────────────────────────────────────────────

    def _tile_key(self, lat: float, lon: float) -> str:
        """SRTM tile key for the 1°×1° tile containing (lat, lon)."""
        tlat = int(math.floor(lat))
        tlon = int(math.floor(lon))
        ns = "N" if tlat >= 0 else "S"
        ew = "E" if tlon >= 0 else "W"
        return f"{ns}{abs(tlat):02d}{ew}{abs(tlon):03d}"

    def _tile_path(self, key: str) -> Optional[str]:
        """Search for the tile file in dem_dir."""
        for ext in (".hgt", ".HGT"):
            p = os.path.join(self._dem_dir, key + ext)
            if os.path.isfile(p):
                return p
        # Also try subdirectory structure: dem_dir/N34/N34W118.hgt
        for ext in (".hgt", ".HGT"):
            subdir = key[:3]
            p = os.path.join(self._dem_dir, subdir, key + ext)
            if os.path.isfile(p):
                return p
        return None

    def _load_tile(self, lat: float, lon: float):
        """Load (and cache) the SRTM tile for (lat, lon)."""
        key = self._tile_key(lat, lon)
        if key in self._cache:
            return self._cache[key]

        path = self._tile_path(key)
        if path is None:
            logger.debug(f"DEM tile not found: {key} (elevation will be 0)")
            self._cache[key] = None
            return None

        try:
            data = np.fromfile(path, dtype=">i2")  # big-endian int16
            n = int(round(math.sqrt(len(data))))  # 3601 for SRTM1, 1201 for SRTM3
            tile = data.reshape((n, n)).astype(np.float32)
            # Replace void values (-32768) with 0
            tile[tile == -32768] = 0.0
            self._cache[key] = tile
            logger.info(f"DEM tile loaded: {key} ({n}×{n}, path={path})")
            return tile
        except Exception as e:
            logger.warning(f"Failed to load DEM tile {path}: {e}")
            self._cache[key] = None
            return None

    def _sample(self, tile, lat: float, lon: float) -> float:
        """Bilinear interpolation of elevation at (lat, lon) from tile array."""
        n = tile.shape[0]
        # n-1 intervals spanning 1 degree
        tlat = math.floor(lat)
        tlon = math.floor(lon)
        # Fractional position within tile (0 = SW corner, 1 = NE corner)
        fx = (lon - tlon) * (n - 1)
        fy = (lat - tlat) * (n - 1)
        # SRTM tiles are stored N→S, so row 0 is the northern edge
        row = (n - 1) - fy
        col = fx

        r0, c0 = int(row), int(col)
        r1 = min(r0 + 1, n - 1)
        c1 = min(c0 + 1, n - 1)
        dr = row - r0
        dc = col - c0

        v00 = float(tile[r0, c0])
        v01 = float(tile[r0, c1])
        v10 = float(tile[r1, c0])
        v11 = float(tile[r1, c1])

        return (
            v00 * (1 - dc) * (1 - dr)
            + v01 * dc * (1 - dr)
            + v10 * (1 - dc) * dr
            + v11 * dc * dr
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_default_provider: Optional[DEMProvider] = None


def get_provider() -> DEMProvider:
    """Return the module-level DEMProvider (lazy init)."""
    global _default_provider
    if _default_provider is None:
        _default_provider = DEMProvider(os.getenv("DEM_DIR", "/data/dem"))
    return _default_provider


# ── Helpers ──────────────────────────────────────────────────────────────────


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
