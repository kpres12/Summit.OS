"""
World Model HMAC — Heli.OS Security

Signs world-model state snapshots with HMAC-SHA256 before forwarding to
downstream services (intelligence, tasking). Recipients verify the HMAC before
trusting the snapshot. Prevents injection of fabricated world state.

Key: SUMMIT_WM_HMAC_KEY env var (hex-encoded 32 bytes). Auto-generated and
logged if missing (dev mode).
"""

import hashlib
import hmac
import json
import logging
import os
import secrets
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_key_cache: Optional[bytes] = None
_warned_ephemeral = False


def get_key() -> bytes:
    """
    Return the HMAC key.

    Reads SUMMIT_WM_HMAC_KEY (hex-encoded 32 bytes) from the environment.
    If absent, generates an ephemeral key and logs a warning once.
    """
    global _key_cache, _warned_ephemeral

    if _key_cache is not None:
        return _key_cache

    raw = os.environ.get("SUMMIT_WM_HMAC_KEY")
    if raw:
        try:
            key = bytes.fromhex(raw)
            if len(key) != 32:
                raise ValueError(f"Expected 32 bytes, got {len(key)}")
            _key_cache = key
            return _key_cache
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "world_model_hmac: invalid SUMMIT_WM_HMAC_KEY (%s) — generating ephemeral key",
                exc,
            )

    # Generate ephemeral key for dev mode
    _key_cache = secrets.token_bytes(32)
    if not _warned_ephemeral:
        logger.warning(
            "world_model_hmac: SUMMIT_WM_HMAC_KEY not set — using ephemeral key "
            "(snapshots will NOT verify across restarts). Set SUMMIT_WM_HMAC_KEY in production."
        )
        _warned_ephemeral = True

    return _key_cache


def _canonical(snapshot: dict) -> bytes:
    """Return deterministic JSON bytes for *snapshot* (sort_keys=True)."""
    return json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_snapshot(snapshot: dict) -> str:
    """
    Compute HMAC-SHA256 over the canonical JSON of *snapshot*.
    Returns the hex digest string.
    """
    key = get_key()
    digest = hmac.new(key, _canonical(snapshot), hashlib.sha256).hexdigest()
    return digest


def verify_snapshot(snapshot: dict, digest: str) -> bool:
    """
    Verify that *digest* matches the HMAC-SHA256 of *snapshot*.
    Uses constant-time comparison to prevent timing attacks.
    """
    expected = sign_snapshot(snapshot)
    try:
        return hmac.compare_digest(expected, digest)
    except (TypeError, ValueError):
        return False


def attach_hmac(snapshot: dict) -> dict:
    """
    Return a copy of *snapshot* with an added ``_hmac`` field containing
    the HMAC-SHA256 digest of the original snapshot (without ``_hmac``).
    """
    # Work on a copy without any existing _hmac to get a stable digest
    clean = {k: v for k, v in snapshot.items() if k != "_hmac"}
    digest = sign_snapshot(clean)
    result = dict(clean)
    result["_hmac"] = digest
    return result


def strip_and_verify(snapshot: dict) -> Tuple[dict, bool]:
    """
    Pop the ``_hmac`` field from *snapshot*, verify it, and return
    (clean_snapshot, valid).

    *snapshot* is not mutated — a clean copy is returned.
    Returns (snapshot_without_hmac, False) if ``_hmac`` is missing or invalid.
    """
    snapshot_copy = dict(snapshot)
    digest = snapshot_copy.pop("_hmac", None)

    if digest is None:
        logger.warning("strip_and_verify: snapshot missing _hmac field")
        return (snapshot_copy, False)

    valid = verify_snapshot(snapshot_copy, digest)
    if not valid:
        logger.warning("strip_and_verify: HMAC verification FAILED — snapshot may be tampered")

    return (snapshot_copy, valid)
