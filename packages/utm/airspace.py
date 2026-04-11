"""
packages/utm/airspace.py — Combined airspace checker for Summit.OS.

This is the single entry point used by the tasking service before any
mission is submitted to OPA. It:

  1. Queries FAA UAS Facility Maps for the authorized altitude at mission center
  2. Fetches active NOTAMs and converts restriction NOTAMs to OPA exclusion zones
  3. Determines if LAANC authorization is required (facility map altitude == 0)
  4. Optionally submits LAANC authorization and returns USS response

The `AirspaceResult` returned is directly injected into the OPA `evaluate_geofence()`
call as the `geofences` list, and used to set `max_altitude_m`.

Usage in tasking service:
    from packages.utm.airspace import AirspaceChecker

    checker = AirspaceChecker()
    result = await checker.check(
        lat=37.77, lon=-122.41, radius_nm=3,
        waypoints=[{"lat": 37.77, "lon": -122.41, "alt": 50}]
    )

    # Inject into OPA
    opa_result = await opa_client.evaluate_geofence(
        asset_id=asset_id,
        waypoints=waypoints,
        geofences=result.geofences,
        max_altitude_m=result.authorized_altitude_m,
    )

    # Surface to operator
    if result.laanc_required and not result.laanc_authorized:
        raise HTTPException(400, detail={
            "airspace_violation": "LAANC authorization required",
            "laanc_state": result.laanc_state,
        })
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("utm.airspace")

# ft → metres conversion factor
_FT_PER_M = 3.28084
_M_PER_FT = 1.0 / _FT_PER_M


@dataclass
class AirspaceResult:
    """
    Unified airspace status for a mission area.

    geofences           — OPA-compatible exclusion/advisory zones (NOTAMs + facility map)
    authorized_altitude_ft — Part 107 authorized altitude at mission center (ft AGL)
    authorized_altitude_m  — same in metres (for OPA max_altitude_m check)
    laanc_required      — True if mission center is in Class B/C/D with 0ft ceiling
    laanc_authorized    — True if LAANC response came back authorized
    laanc_state         — "AUTHORIZED" | "NOT_AUTHORIZED" | "STUB" | "SKIPPED"
    laanc_conditions    — any conditions placed on the authorization
    notam_count         — total active NOTAMs in the area
    restriction_count   — NOTAMs that converted to OPA exclusion zones
    facility_cells      — number of facility map cells returned
    airspace_class      — airspace class at mission center (if known)
    airport_id          — ICAO identifier of controlling airport (if any)
    errors              — non-fatal errors encountered during fetch
    """
    geofences: List[Dict[str, Any]] = field(default_factory=list)
    authorized_altitude_ft: int = 400
    authorized_altitude_m: float = 121.92        # 400ft in metres
    laanc_required: bool = False
    laanc_authorized: bool = False
    laanc_state: str = "SKIPPED"
    laanc_conditions: List[str] = field(default_factory=list)
    notam_count: int = 0
    restriction_count: int = 0
    facility_cells: int = 0
    airspace_class: Optional[str] = None
    airport_id: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"alt={self.authorized_altitude_ft}ft",
            f"notams={self.notam_count}",
            f"restrictions={self.restriction_count}",
            f"cells={self.facility_cells}",
        ]
        if self.laanc_required:
            parts.append(f"laanc={self.laanc_state}")
        if self.airspace_class:
            parts.append(f"class={self.airspace_class}")
        return " | ".join(parts)


class AirspaceChecker:
    """
    Summit.OS airspace awareness layer.

    Aggregates FAA NOTAM data, facility map data, and LAANC authorization
    into a single `AirspaceResult` ready for OPA injection.

    All external API calls fail-open (errors are recorded in result.errors
    but do not prevent the mission from proceeding — OPA handles final deny).

    Parameters:
        notam_radius_nm     — NOTAM search radius (default: 25nm)
        facility_radius_nm  — Facility map search radius (default: 5nm)
        auto_laanc          — Submit LAANC if required (default: True)
        operator_id         — FAA operator ID (required for live LAANC)
        uas_serial          — UAS serial number (required for live LAANC)
    """

    def __init__(
        self,
        notam_radius_nm: float = 25.0,
        facility_radius_nm: float = 5.0,
        auto_laanc: bool = True,
        operator_id: str = "SUMMIT_OPS",
        uas_serial: str = "SUMMIT_UAS_001",
    ) -> None:
        self._notam_radius = notam_radius_nm
        self._facility_radius = facility_radius_nm
        self._auto_laanc = auto_laanc
        self._operator_id = operator_id
        self._uas_serial = uas_serial

        # Lazy-load clients to avoid import errors at module load time
        self._notam_client = None
        self._facility_client = None
        self._laanc_client = None

    def _get_notam_client(self):
        if self._notam_client is None:
            from .notam import NotamClient
            self._notam_client = NotamClient()
        return self._notam_client

    def _get_facility_client(self):
        if self._facility_client is None:
            from .facility_map import FacilityMapClient
            self._facility_client = FacilityMapClient()
        return self._facility_client

    def _get_laanc_client(self):
        if self._laanc_client is None:
            from .laanc import LaancClient
            self._laanc_client = LaancClient()
        return self._laanc_client

    async def check(
        self,
        lat: float,
        lon: float,
        radius_nm: float = 3.0,
        waypoints: Optional[List[Dict[str, float]]] = None,
        max_altitude_ft: Optional[float] = None,
    ) -> AirspaceResult:
        """
        Full airspace check for a mission area.

        Args:
            lat, lon        — mission center (decimal degrees)
            radius_nm       — mission radius (nautical miles)
            waypoints       — list of {"lat": float, "lon": float} or
                              {"lat", "lon", "alt"} dicts for LAANC polygon
            max_altitude_ft — requested mission ceiling (ft AGL)
                              defaults to 400 (FAA Part 107 max)

        Returns AirspaceResult with geofences, altitude limits, LAANC status.
        """
        result = AirspaceResult()
        requested_alt = max_altitude_ft or 400.0

        # Run facility map + NOTAM fetches concurrently in thread pool
        # (both use blocking urllib — wrap in asyncio executor)
        loop = asyncio.get_event_loop()

        facility_future = loop.run_in_executor(
            None, self._fetch_facility, lat, lon
        )
        notam_future = loop.run_in_executor(
            None, self._fetch_notams, lat, lon
        )

        facility_result, notam_result = await asyncio.gather(
            facility_future, notam_future, return_exceptions=True
        )

        # ── Facility map ──────────────────────────────────────────────────────
        if isinstance(facility_result, Exception):
            result.errors.append(f"facility_map: {facility_result}")
            logger.warning("Facility map fetch failed: %s", facility_result)
        else:
            cells, facility_geofences = facility_result
            result.facility_cells = len(cells)

            # Find the cell containing the mission center
            for cell in cells:
                if (cell.min_lat <= lat <= cell.max_lat and
                        cell.min_lon <= lon <= cell.max_lon):
                    result.authorized_altitude_ft = cell.authorized_altitude_ft
                    result.authorized_altitude_m = cell.authorized_altitude_ft * _M_PER_FT
                    result.laanc_required = cell.laanc_required
                    result.airspace_class = cell.airspace_class
                    result.airport_id = cell.airport_id
                    break

            result.geofences.extend(facility_geofences)

        # ── NOTAMs ────────────────────────────────────────────────────────────
        if isinstance(notam_result, Exception):
            result.errors.append(f"notam: {notam_result}")
            logger.warning("NOTAM fetch failed: %s", notam_result)
        else:
            notams, notam_geofences = notam_result
            result.notam_count = len(notams)
            result.restriction_count = len(notam_geofences)
            result.geofences.extend(notam_geofences)

        # ── LAANC authorization ───────────────────────────────────────────────
        if result.laanc_required and self._auto_laanc:
            try:
                laanc_resp = await loop.run_in_executor(
                    None,
                    self._submit_laanc,
                    waypoints or [{"lat": lat, "lon": lon}],
                    requested_alt,
                )
                result.laanc_state = laanc_resp.state
                result.laanc_authorized = laanc_resp.is_authorized
                result.laanc_conditions = laanc_resp.conditions

                if laanc_resp.is_authorized and laanc_resp.authorized_altitude_ft is not None:
                    # LAANC may cap altitude below requested
                    authorized = laanc_resp.authorized_altitude_ft
                    result.authorized_altitude_ft = int(authorized)
                    result.authorized_altitude_m = authorized * _M_PER_FT

                    if authorized < requested_alt:
                        # Add an advisory geofence noting the altitude cap
                        result.geofences.append({
                            "geofence_id": f"laanc-{laanc_resp.operation_id}",
                            "type": "advisory",
                            "source": "laanc",
                            "description": (
                                f"LAANC authorized to {authorized}ft AGL "
                                f"(requested {requested_alt}ft)"
                            ),
                            "authorized_altitude_ft": authorized,
                            "laanc_state": laanc_resp.state,
                            "conditions": laanc_resp.conditions,
                        })
                elif laanc_resp.is_denied:
                    # Add an exclusion geofence for the denied area
                    result.geofences.append({
                        "geofence_id": f"laanc-denied-{laanc_resp.operation_id}",
                        "type": "exclusion",
                        "source": "laanc",
                        "description": (
                            f"LAANC authorization denied: {laanc_resp.message}"
                        ),
                        "authorized_altitude_ft": 0,
                        "laanc_state": "NOT_AUTHORIZED",
                    })

            except Exception as exc:
                result.errors.append(f"laanc: {exc}")
                logger.warning("LAANC authorization failed: %s", exc)
                # Fail-open: LAANC error does not block mission — OPA decides
                result.laanc_state = "ERROR"
        elif result.laanc_required and not self._auto_laanc:
            result.laanc_state = "REQUIRED_NOT_SUBMITTED"
        else:
            result.laanc_state = "NOT_REQUIRED"
            result.laanc_authorized = True  # No LAANC needed → implicitly authorized

        logger.info(
            "Airspace check complete: %s (%.4f,%.4f r=%.1fnm)",
            result.summary(), lat, lon, radius_nm,
        )
        return result

    def _fetch_facility(self, lat: float, lon: float):
        """Blocking — runs in thread pool."""
        try:
            client = self._get_facility_client()
            cells = client.fetch_cells(lat, lon, radius_nm=self._facility_radius)
            geofences = client.exclusion_geofences(lat, lon, radius_nm=self._facility_radius)
            return cells, geofences
        except Exception as exc:
            raise exc

    def _fetch_notams(self, lat: float, lon: float):
        """Blocking — runs in thread pool."""
        try:
            client = self._get_notam_client()
            notams = client.fetch(lat, lon, radius_nm=self._notam_radius)
            geofences = client.restriction_geofences(lat, lon, radius_nm=self._notam_radius)
            return notams, geofences
        except Exception as exc:
            raise exc

    def _submit_laanc(
        self,
        waypoints: List[Dict[str, float]],
        max_altitude_ft: float,
    ):
        """Blocking — runs in thread pool."""
        client = self._get_laanc_client()
        return client.authorize_for_mission(
            operator_id=self._operator_id,
            uas_serial=self._uas_serial,
            waypoints=waypoints,
            max_altitude_ft=max_altitude_ft,
        )

    def check_sync(
        self,
        lat: float,
        lon: float,
        radius_nm: float = 3.0,
        waypoints: Optional[List[Dict[str, float]]] = None,
        max_altitude_ft: Optional[float] = None,
    ) -> AirspaceResult:
        """
        Synchronous wrapper for use outside of async contexts.
        Creates a new event loop if none is running.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context — caller should use await check()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.check(lat, lon, radius_nm, waypoints, max_altitude_ft),
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.check(lat, lon, radius_nm, waypoints, max_altitude_ft)
                )
        except RuntimeError:
            return asyncio.run(
                self.check(lat, lon, radius_nm, waypoints, max_altitude_ft)
            )
