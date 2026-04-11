"""
packages/utm — Unmanned Traffic Management integration for Summit.OS.

Provides real-time airspace awareness by pulling from public FAA data sources
and structuring the results as OPA-compatible geofence objects that the tasking
service injects into every policy evaluation.

Components:
  notam       — FAA NOTAM API client (active restrictions, TFRs)
  facility_map— FAA UAS Facility Maps (Part 107 authorized altitudes per grid)
  laanc       — LAANC authorization request/response (ASTM F3548-21)
  airspace    — Combined checker used by the tasking service

Quickstart:
  from packages.utm.airspace import AirspaceChecker
  checker = AirspaceChecker()
  result = await checker.check(lat=37.77, lon=-122.41, radius_nm=3, waypoints=[...])
  # result.geofences → list of OPA-compatible exclusion/inclusion dicts
  # result.authorized_altitude_ft → max Part 107 altitude at mission center
  # result.laanc_required → True if LAANC authorization needed
  # result.notams → list of active NOTAM summaries
"""

from .airspace import AirspaceChecker, AirspaceResult
from .notam import NotamClient, Notam
from .facility_map import FacilityMapClient, FacilityCell
from .laanc import LaancClient, LaancRequest, LaancResponse

__all__ = [
    "AirspaceChecker",
    "AirspaceResult",
    "NotamClient",
    "Notam",
    "FacilityMapClient",
    "FacilityCell",
    "LaancClient",
    "LaancRequest",
    "LaancResponse",
]
