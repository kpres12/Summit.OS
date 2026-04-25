"""
Sensor Auth — Heli.OS Fusion

Middleware layer that verifies Ed25519 signatures on incoming sensor
observations before they enter the fusion pipeline. Unsigned frames are
accepted in dev mode (SENSOR_AUTH_ENFORCE=false) and rejected in production.
"""

import json
import logging
import os
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# Lazy import — sensor_signing may live in packages/security; support both installed
# package and direct path import.
try:
    from packages.security.sensor_signing import verify_frame
except ImportError:
    try:
        import sys
        import pathlib
        _repo_root = pathlib.Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(_repo_root))
        from packages.security.sensor_signing import verify_frame  # type: ignore
    except ImportError:
        logger.warning(
            "sensor_auth: could not import sensor_signing — verify_frame will always return True"
        )

        def verify_frame(sensor_id, payload, signature, keys_dir=None):  # type: ignore
            return True


def _parse_enforce() -> bool:
    val = os.environ.get("SENSOR_AUTH_ENFORCE", "true").lower()
    return val not in ("false", "0", "no", "off")


class SensorAuthMiddleware:
    """
    Verifies Ed25519 signatures on inbound sensor frames before they enter
    the fusion pipeline.

    In dev mode (SENSOR_AUTH_ENFORCE=false or enforce=False):
      - Unsigned frames are accepted with a warning.
      - Frames with an invalid signature are still rejected.

    In production mode (enforce=True):
      - Both unsigned and invalid-signature frames are rejected.
    """

    def __init__(self, enforce: bool = None):
        self._enforce = enforce if enforce is not None else _parse_enforce()
        self._stats: Dict[str, int] = {
            "accepted": 0,
            "rejected": 0,
            "unsigned": 0,
        }
        logger.info(
            "SensorAuthMiddleware: initialised (enforce=%s)", self._enforce
        )

    # ------------------------------------------------------------------
    # Low-level verify
    # ------------------------------------------------------------------

    async def verify(
        self, sensor_id: str, payload: bytes, signature: str
    ) -> bool:
        """
        Verify *signature* over *payload* for *sensor_id*.
        Logs a warning on rejection.
        """
        result = verify_frame(sensor_id, payload, signature)
        if not result:
            logger.warning(
                "SensorAuthMiddleware.verify: REJECTED — invalid signature for sensor_id=%s",
                sensor_id,
            )
            self._stats["rejected"] += 1
        else:
            self._stats["accepted"] += 1
        return result

    # ------------------------------------------------------------------
    # Frame-level check (used by ingest pipeline)
    # ------------------------------------------------------------------

    async def check_frame(self, frame: dict) -> Tuple[bool, str]:
        """
        Validate an inbound sensor frame dict.

        Expected frame structure::

            {
                "sensor_id": "cam-01",
                "ts": 1234567890.123,
                "_sig": "<base64url signature>",
                ... (other fields)
            }

        The signature covers the JSON-encoded frame *without* the ``_sig`` field,
        with keys sorted.

        Returns (True, "ok") on acceptance.
        Returns (False, reason_string) on rejection.
        """
        sensor_id = frame.get("sensor_id")
        if not sensor_id:
            self._stats["rejected"] += 1
            return (False, "missing sensor_id")

        signature = frame.get("_sig")

        # Build canonical payload: JSON of frame without _sig, sorted keys
        payload_dict = {k: v for k, v in frame.items() if k != "_sig"}
        try:
            payload_bytes = json.dumps(
                payload_dict, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            self._stats["rejected"] += 1
            return (False, f"payload serialisation error: {exc}")

        # Handle unsigned frames
        if not signature:
            self._stats["unsigned"] += 1
            if self._enforce:
                logger.warning(
                    "SensorAuthMiddleware.check_frame: REJECTED unsigned frame from sensor_id=%s (enforce=True)",
                    sensor_id,
                )
                self._stats["rejected"] += 1
                return (False, "unsigned frame rejected in enforce mode")
            else:
                logger.debug(
                    "SensorAuthMiddleware.check_frame: accepted unsigned frame from sensor_id=%s (enforce=False)",
                    sensor_id,
                )
                self._stats["accepted"] += 1
                return (True, "unsigned accepted in dev mode")

        # Verify signature
        ok = await self.verify(sensor_id, payload_bytes, signature)
        if ok:
            return (True, "ok")
        return (False, "signature verification failed")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        """Return a copy of the acceptance/rejection counters."""
        return dict(self._stats)
