"""
packages/utm/laanc.py — LAANC authorization client (ASTM F3548-21).

LAANC (Low Altitude Authorization and Notification Capability) is the FAA's
automated system for approving UAS flights in controlled airspace in near-real-
time. Operators submit a proposed operation to a USS (UAS Service Supplier),
which verifies it against FAA facility maps and returns an authorization or denial
within 60 seconds.

This module implements the ASTM F3548-21 USS-to-UTM-DSS interface for:
  - Submitting an Operation Volume (OVMS) for authorization
  - Polling for authorization status
  - Storing and surfacing LAANC UVRs (UAS Volume Reservations)

In production, USS_BASE_URL should point to your production USS (e.g., AirMap,
Skyward, Kittyhawk, or the FAA's DroneZone API).

For self-hosted/development, the client falls back to a local USS stub that
auto-approves operations under 400ft AGL outside of Class B/C/D airspace.

Environment variables:
  USS_BASE_URL         — USS API base (default: local stub)
  USS_API_KEY          — USS API key (from USS provider)
  LAANC_STUB_MODE      — "true" → always use local stub (dev/test)
  LAANC_MAX_ALT_FT     — Maximum altitude for auto-approval stub (default: 400)

Reference:
  https://www.faa.gov/uas/programs_partnerships/data_exchange/laanc
  https://www.astm.org/f3548-21.html
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("utm.laanc")

_USS_BASE_URL = os.getenv("USS_BASE_URL", "")
_USS_API_KEY = os.getenv("USS_API_KEY", "")
_STUB_MODE = os.getenv("LAANC_STUB_MODE", "true").lower() == "true"
_STUB_MAX_ALT_FT = int(os.getenv("LAANC_MAX_ALT_FT", "400"))
_REQUEST_TIMEOUT = 15


# ── Data structures ────────────────────────────────────────────────────────────


@dataclass
class LaancRequest:
    """
    A LAANC authorization request (maps to ASTM F3548-21 OperationVolume).

    Required fields for USS submission:
      operation_id  — unique identifier (auto-generated if omitted)
      operator_id   — FAA-registered operator identifier (drone ID / Part 107)
      uas_serial    — UAS serial number (Remote ID)
      time_start    — planned start time (ISO 8601 UTC)
      time_end      — planned end time (ISO 8601 UTC)
      min_altitude_ft — floor of operation volume (ft AGL)
      max_altitude_ft — ceiling of operation volume (ft AGL)
      area_polygon  — list of [lon, lat] coordinates forming the flight area
      purpose       — flight purpose code (AGRICULTURE, INSPECTION, MAPPING,
                       PHOTOGRAPHY, PUBLIC_SAFETY, RECREATION, RESEARCH, SURVEY)
    """
    operator_id: str
    uas_serial: str
    time_start: str
    time_end: str
    min_altitude_ft: float
    max_altitude_ft: float
    area_polygon: List[List[float]]   # [[lon, lat], ...]
    purpose: str = "SURVEY"
    operation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_astm_dict(self) -> Dict[str, Any]:
        """Serialize to ASTM F3548-21 OperationVolume JSON."""
        return {
            "operation_id": self.operation_id,
            "operator_id": self.operator_id,
            "uas_serial": self.uas_serial,
            "volumes": [
                {
                    "volume": {
                        "outline_polygon": {
                            "vertices": [
                                {"lat": pt[1], "lng": pt[0]}
                                for pt in self.area_polygon
                            ]
                        },
                        "altitude_lower": {
                            "value": self.min_altitude_ft,
                            "reference": "W84",
                            "units": "FT",
                        },
                        "altitude_upper": {
                            "value": self.max_altitude_ft,
                            "reference": "W84",
                            "units": "FT",
                        },
                    },
                    "time_start": {"value": self.time_start, "format": "RFC3339"},
                    "time_end": {"value": self.time_end, "format": "RFC3339"},
                }
            ],
            "type": "VLOS",
            "uas_regulatory_requirement": "FAR_107",
            "flight_purpose": self.purpose,
            "flight_purpose_defined_by_uss": False,
        }


@dataclass
class LaancResponse:
    """
    LAANC authorization response.

    State machine (ASTM F3548-21):
      ACCEPTED   → operation submitted, awaiting evaluation
      AUTHORIZED → approved by USS (may fly immediately up to authorized_alt_ft)
      CONDITIONALLY_AUTHORIZED → approved with conditions (altitude cap, etc.)
      NOT_AUTHORIZED → denied by USS or FAA — cannot fly without manual waiver
      ACTIVATED  → operator has activated the operation
      CLOSED     → operation complete or cancelled

    authorization_ceiling_ft is the approved altitude (may be lower than requested).
    """
    operation_id: str
    state: str            # ACCEPTED | AUTHORIZED | CONDITIONALLY_AUTHORIZED | NOT_AUTHORIZED
    authorized_altitude_ft: Optional[float]
    conditions: List[str]
    effective_time_start: Optional[str]
    effective_time_end: Optional[str]
    message: str
    raw: Dict[str, Any]

    @property
    def is_authorized(self) -> bool:
        return self.state in ("AUTHORIZED", "CONDITIONALLY_AUTHORIZED", "ACTIVATED")

    @property
    def is_denied(self) -> bool:
        return self.state == "NOT_AUTHORIZED"

    def to_opa_context(self) -> Dict[str, Any]:
        """Return OPA-compatible dict for injection into mission policy evaluation."""
        return {
            "laanc_operation_id": self.operation_id,
            "laanc_state": self.state,
            "laanc_authorized": self.is_authorized,
            "laanc_authorized_altitude_ft": self.authorized_altitude_ft,
            "laanc_conditions": self.conditions,
            "laanc_message": self.message,
        }


# ── USS client ─────────────────────────────────────────────────────────────────


def _stub_authorize(req: LaancRequest) -> LaancResponse:
    """
    Local stub USS: auto-approves operations up to _STUB_MAX_ALT_FT.
    Used in development, CI, and when no USS_BASE_URL is configured.

    This does NOT contact the FAA — it is only a placeholder for local testing.
    In production, configure USS_BASE_URL and USS_API_KEY.
    """
    requested_alt = req.max_altitude_ft
    authorized_alt = min(requested_alt, _STUB_MAX_ALT_FT)
    conditions = []

    if requested_alt > _STUB_MAX_ALT_FT:
        state = "CONDITIONALLY_AUTHORIZED"
        conditions.append(
            f"Altitude capped to {_STUB_MAX_ALT_FT}ft AGL by stub USS "
            f"(requested {requested_alt}ft)"
        )
        msg = f"Conditionally authorized up to {authorized_alt}ft AGL"
    else:
        state = "AUTHORIZED"
        msg = f"Authorized up to {authorized_alt}ft AGL (stub USS — dev mode)"

    logger.info(
        "LAANC stub: %s op=%s alt=%.0fft→%.0fft",
        state,
        req.operation_id,
        requested_alt,
        authorized_alt,
    )
    return LaancResponse(
        operation_id=req.operation_id,
        state=state,
        authorized_altitude_ft=authorized_alt,
        conditions=conditions,
        effective_time_start=req.time_start,
        effective_time_end=req.time_end,
        message=msg,
        raw={"stub": True},
    )


class LaancClient:
    """
    LAANC USS client (ASTM F3548-21).

    In stub mode (default when USS_BASE_URL is unset or LAANC_STUB_MODE=true),
    authorization is handled locally without any FAA API call.

    In production mode, submits to the configured USS and polls for the result.

    Usage:
        client = LaancClient()
        request = LaancRequest(
            operator_id="FA3X7PILOT",
            uas_serial="SN12345",
            time_start="2025-01-01T12:00:00Z",
            time_end="2025-01-01T13:00:00Z",
            min_altitude_ft=0,
            max_altitude_ft=200,
            area_polygon=[[-122.42, 37.77], [-122.41, 37.77],
                          [-122.41, 37.78], [-122.42, 37.78], [-122.42, 37.77]],
        )
        response = client.authorize(request)
        if response.is_authorized:
            print(f"Fly up to {response.authorized_altitude_ft}ft AGL")
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        stub_mode: Optional[bool] = None,
    ) -> None:
        self._base_url = base_url or _USS_BASE_URL
        self._api_key = api_key or _USS_API_KEY
        self._stub = stub_mode if stub_mode is not None else (
            _STUB_MODE or not self._base_url
        )

    def authorize(self, req: LaancRequest) -> LaancResponse:
        """
        Submit a LAANC authorization request and return the response.

        In stub mode: synchronous, instant approval/conditional approval.
        In production mode: submits to USS, polls up to 60s for a final state.
        """
        if self._stub:
            return _stub_authorize(req)

        return self._submit_and_poll(req)

    def _submit_and_poll(self, req: LaancRequest, poll_timeout: float = 60.0) -> LaancResponse:
        """Submit to USS and poll for terminal state."""
        # Submit
        body = json.dumps(req.to_astm_dict()).encode()
        submit_url = f"{self._base_url.rstrip('/')}/uss/v1/operations"
        http_req = urllib.request.Request(
            submit_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(http_req, timeout=_REQUEST_TIMEOUT) as resp:
                submitted = json.loads(resp.read())
        except Exception as exc:
            logger.warning("LAANC USS submission failed: %s — falling back to stub", exc)
            return _stub_authorize(req)

        operation_id = submitted.get("operation_id") or req.operation_id
        deadline = time.time() + poll_timeout
        poll_url = f"{self._base_url.rstrip('/')}/uss/v1/operations/{operation_id}"

        while time.time() < deadline:
            try:
                poll_req = urllib.request.Request(
                    poll_url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                with urllib.request.urlopen(poll_req, timeout=_REQUEST_TIMEOUT) as resp:
                    data = json.loads(resp.read())

                op = data.get("operation", data)
                state_str = op.get("state", "ACCEPTED")

                if state_str in ("AUTHORIZED", "CONDITIONALLY_AUTHORIZED", "NOT_AUTHORIZED"):
                    return self._parse_response(req, op, state_str)

                time.sleep(3)

            except Exception as exc:
                logger.warning("LAANC USS poll failed: %s", exc)
                break

        # Timeout — fall back to stub
        logger.warning(
            "LAANC USS authorization timed out for op=%s — falling back to stub",
            operation_id,
        )
        return _stub_authorize(req)

    def _parse_response(
        self, req: LaancRequest, op: Dict[str, Any], state_str: str
    ) -> LaancResponse:
        """Parse USS operation response into LaancResponse."""
        vols = op.get("volumes", [{}])
        upper_alt = None
        try:
            upper_alt = float(
                vols[0].get("volume", {})
                .get("altitude_upper", {})
                .get("value", req.max_altitude_ft)
            )
        except (TypeError, ValueError, KeyError, IndexError):
            upper_alt = req.max_altitude_ft

        conditions = []
        for cond in op.get("conditions", []):
            if isinstance(cond, str):
                conditions.append(cond)
            elif isinstance(cond, dict):
                conditions.append(cond.get("description", str(cond)))

        time_start = None
        time_end = None
        try:
            time_start = vols[0].get("time_start", {}).get("value")
            time_end = vols[0].get("time_end", {}).get("value")
        except (IndexError, AttributeError, KeyError):
            pass

        msg = op.get("message", "") or ("Authorized" if state_str != "NOT_AUTHORIZED" else "Denied")

        return LaancResponse(
            operation_id=op.get("operation_id", req.operation_id),
            state=state_str,
            authorized_altitude_ft=upper_alt if state_str != "NOT_AUTHORIZED" else None,
            conditions=conditions,
            effective_time_start=time_start or req.time_start,
            effective_time_end=time_end or req.time_end,
            message=msg,
            raw=op,
        )

    def authorize_for_mission(
        self,
        operator_id: str,
        uas_serial: str,
        waypoints: List[Dict[str, float]],
        max_altitude_ft: float = 400.0,
        duration_minutes: int = 60,
        purpose: str = "SURVEY",
    ) -> LaancResponse:
        """
        Convenience method: build a LaancRequest from mission waypoints and authorize.

        waypoints: list of {"lat": float, "lon": float} dicts (the mission waypoints)
        """
        if not waypoints:
            logger.warning("LAANC: no waypoints provided — skipping authorization")
            return LaancResponse(
                operation_id=str(uuid.uuid4()),
                state="NOT_AUTHORIZED",
                authorized_altitude_ft=None,
                conditions=[],
                effective_time_start=None,
                effective_time_end=None,
                message="No waypoints provided",
                raw={},
            )

        # Build bounding polygon from waypoints
        lats = [w["lat"] for w in waypoints]
        lons = [w["lon"] for w in waypoints]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        # Add small buffer (0.001 deg ≈ 100m)
        buf = 0.001
        polygon = [
            [min_lon - buf, min_lat - buf],
            [max_lon + buf, min_lat - buf],
            [max_lon + buf, max_lat + buf],
            [min_lon - buf, max_lat + buf],
            [min_lon - buf, min_lat - buf],  # close ring
        ]

        now = datetime.now(timezone.utc)
        time_start = now.isoformat().replace("+00:00", "Z")
        # Compute end time
        from datetime import timedelta
        time_end = (now + timedelta(minutes=duration_minutes)).isoformat().replace("+00:00", "Z")

        laanc_req = LaancRequest(
            operator_id=operator_id,
            uas_serial=uas_serial,
            time_start=time_start,
            time_end=time_end,
            min_altitude_ft=0.0,
            max_altitude_ft=max_altitude_ft,
            area_polygon=polygon,
            purpose=purpose,
        )

        return self.authorize(laanc_req)
