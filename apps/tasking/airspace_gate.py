"""
apps/tasking/airspace_gate.py — Real-time airspace gate for mission creation.

Called by _validate_policies() in planning.py before OPA policy evaluation.
Fetches live FAA airspace data (NOTAMs + facility map + LAANC) and injects
the resulting geofences into the OPA geofence check.

This is the regulatory enforcement layer — it turns Heli.OS from a system
that *can* fly anywhere into one that *knows* where it's legally allowed to fly.

Integration diagram:
  MissionCreateRequest
       ↓
  airspace_gate.check_mission_airspace()
       ↓                              ↓
  AirspaceChecker.check()      injects geofences
  (NOTAM + FacilityMap + LAANC)      ↓
       ↓                     OPAClient.evaluate_geofence()
  AirspaceResult                     ↓
       ↓                    geofence_violations (list[str])
  violations if:
    - LAANC denied in non-stub mode
    - Waypoints inside NOTAM TFR
    - Altitude exceeds authorized ceiling
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tasking.airspace")

# Path to packages/ — support both containerized and local dev
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Feature flag — allow disabling airspace checks entirely (dev/sim mode)
_AIRSPACE_ENFORCE = os.getenv("AIRSPACE_ENFORCE", "true").lower() == "true"
_AIRSPACE_LAANC_AUTO = os.getenv("AIRSPACE_LAANC_AUTO", "true").lower() == "true"

# Part 107 maximum altitude (ft AGL) — hard ceiling regardless of facility map
_PART_107_MAX_ALT_FT = 400
_PART_107_MAX_ALT_M = _PART_107_MAX_ALT_FT / 3.28084


async def check_mission_airspace(
    req,
    org_id: Optional[str] = None,
    opa_client=None,
) -> List[str]:
    """
    Run the full airspace gate for a mission creation request.

    Args:
        req: MissionCreateRequest (from models.py)
        org_id: organization ID for OPA context
        opa_client: OPAClient instance (optional — creates one if not provided)

    Returns:
        List of violation strings. Empty list = mission is cleared for dispatch.
    """
    if not _AIRSPACE_ENFORCE:
        logger.debug("Airspace enforcement disabled (AIRSPACE_ENFORCE=false)")
        return []

    # Extract mission center and waypoints from request
    center_lat, center_lon, radius_nm, waypoints, max_alt_ft = _extract_mission_geometry(req)

    if center_lat is None or center_lon is None:
        logger.debug("No mission area specified — skipping airspace check")
        return []

    # Run airspace check
    try:
        from packages.utm.airspace import AirspaceChecker
        checker = AirspaceChecker(auto_laanc=_AIRSPACE_LAANC_AUTO)
        airspace = await checker.check(
            lat=center_lat,
            lon=center_lon,
            radius_nm=radius_nm,
            waypoints=waypoints,
            max_altitude_ft=max_alt_ft,
        )
    except ImportError:
        logger.warning(
            "UTM package not available — skipping airspace check. "
            "Install packages/utm to enable real-time airspace awareness."
        )
        return []
    except Exception as exc:
        logger.error("Airspace check failed: %s — fail-open", exc)
        return []

    violations: List[str] = []

    # ── Check 1: LAANC denial (hard block in production, warn in stub mode) ──
    if airspace.laanc_required:
        if airspace.laanc_state == "NOT_AUTHORIZED":
            violations.append(
                "airspace.laanc_denied: LAANC authorization denied for this area — "
                "contact your USS provider or use FAA DroneZone for manual waiver"
            )
        elif airspace.laanc_state in ("REQUIRED_NOT_SUBMITTED", "ERROR"):
            # Warn but don't block — operator may have pre-authorized manually
            logger.warning(
                "LAANC required but not submitted (state=%s) for %.4f,%.4f — "
                "operator must have manual FAA authorization",
                airspace.laanc_state, center_lat, center_lon,
            )
        elif airspace.laanc_state.startswith("AUTHORIZED"):
            logger.info(
                "LAANC authorized: %s (alt≤%sft) for %.4f,%.4f",
                airspace.laanc_state,
                airspace.authorized_altitude_ft,
                center_lat, center_lon,
            )

    # ── Check 2: Requested altitude vs authorized ceiling ──
    if max_alt_ft > airspace.authorized_altitude_ft + 1:  # +1 for float rounding
        violations.append(
            f"airspace.altitude_exceeded: Requested altitude {max_alt_ft:.0f}ft AGL exceeds "
            f"authorized ceiling of {airspace.authorized_altitude_ft}ft AGL at this location"
            + (f" (near {airspace.airport_id})" if airspace.airport_id else "")
            + (f" [Class {airspace.airspace_class}]" if airspace.airspace_class else "")
        )

    # ── Check 3: OPA geofence evaluation with real airspace data ──
    if airspace.geofences and waypoints:
        try:
            if opa_client is None:
                from opa import OPAClient
                opa_client = OPAClient()

            # Normalize waypoints to OPA schema (latitude/longitude/altitude_m)
            opa_waypoints = [
                {
                    "latitude": wp.get("lat", wp.get("latitude", 0.0)),
                    "longitude": wp.get("lon", wp.get("longitude", 0.0)),
                    "altitude_m": wp.get("alt", wp.get("altitude_m", 50.0)),
                }
                for wp in waypoints
            ]

            geofence_result = await opa_client.evaluate_geofence(
                asset_id="pre-flight-check",
                waypoints=opa_waypoints,
                org_id=org_id or "dev",
                geofences=airspace.geofences,
                max_altitude_m=airspace.authorized_altitude_m,
            )

            if not geofence_result.get("allow", True):
                reasons = geofence_result.get("deny_reasons") or []
                for r in reasons:
                    violations.append(f"airspace.geofence: {r}")

                if not reasons:
                    violations.append(
                        "airspace.geofence: Waypoints intersect active airspace restriction"
                    )

        except Exception as exc:
            logger.warning("OPA geofence check failed: %s — skipping geofence gate", exc)

    # Log summary
    if violations:
        logger.warning(
            "Airspace gate BLOCKED mission: %d violations | %s",
            len(violations), airspace.summary(),
        )
    else:
        logger.info(
            "Airspace gate PASSED: %s | restrictions=%d",
            airspace.summary(), len(airspace.geofences),
        )

    return violations


def _extract_mission_geometry(req) -> tuple:
    """
    Extract (center_lat, center_lon, radius_nm, waypoints, max_alt_ft) from
    a MissionCreateRequest. Returns (None, None, ...) if no area specified.
    """
    center_lat = None
    center_lon = None
    radius_nm = 3.0
    waypoints: List[Dict[str, float]] = []
    max_alt_ft = _PART_107_MAX_ALT_FT

    area = getattr(req, "area", None) or {}
    planning_params = getattr(req, "planning_params", None) or {}

    if area:
        center = area.get("center", {})
        center_lat = center.get("lat") or center.get("latitude")
        center_lon = center.get("lon") or center.get("longitude")
        radius_m = area.get("radius_m", 500.0)
        # Convert radius from metres to nautical miles
        radius_nm = max(0.5, radius_m / 1852.0)

        # Build waypoints from polygon or from center
        polygon = area.get("polygon")
        if polygon and isinstance(polygon, list):
            for pt in polygon:
                if isinstance(pt, list) and len(pt) >= 2:
                    waypoints.append({"lat": pt[0], "lon": pt[1]})
                elif isinstance(pt, dict):
                    waypoints.append({
                        "lat": pt.get("lat", pt.get("latitude", 0.0)),
                        "lon": pt.get("lon", pt.get("longitude", 0.0)),
                    })

        if not waypoints and center_lat is not None and center_lon is not None:
            waypoints = [{"lat": center_lat, "lon": center_lon}]

    # Max altitude from planning params
    alt_param = planning_params.get("altitude") or planning_params.get("max_altitude")
    if alt_param is not None:
        try:
            alt_val = float(alt_param)
            # If value looks like metres (>200), convert to feet
            if alt_val > 200:
                max_alt_ft = alt_val * 3.28084
            else:
                max_alt_ft = alt_val
        except (TypeError, ValueError):
            pass

    return center_lat, center_lon, radius_nm, waypoints, max_alt_ft
