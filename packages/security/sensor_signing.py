"""
Sensor Frame Signing — Heli.OS Security

Each sensor adapter signs its observation frames with Ed25519 before publishing
to MQTT. The fusion service verifies signatures before accepting data into the
world model. Unsigned or tampered frames are rejected.

Key management:
  - Keys stored under SUMMIT_SENSOR_KEYS_DIR (default ./sensor_keys)
  - Key file: {sensor_id}.key (private, 32 bytes raw Ed25519 seed)
  - Pub file: {sensor_id}.pub (public key, 32 bytes)
  - generate_keypair(sensor_id) creates both files if absent
"""

import base64
import logging
import os
from pathlib import Path
from typing import Any, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_KEYS_DIR = "./sensor_keys"

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )
    from cryptography.exceptions import InvalidSignature

    _CRYPTO_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CRYPTO_AVAILABLE = False

# In production, the cryptography library MUST be installed. We deliberately
# fail closed: missing crypto = no signing, no verification (rather than
# silent fail-open that lets unsigned frames through). This was previously a
# fail-open bug that allowed engagement-authorization defaults to be bypassed.
# To explicitly opt-in to development pass-through, set:
#   HELI_SECURITY_DEV_FAIL_OPEN=1
# Never set this in production. Audit logs record the dev-mode flag.
_DEV_FAIL_OPEN = os.environ.get("HELI_SECURITY_DEV_FAIL_OPEN") == "1"
if not _CRYPTO_AVAILABLE and not _DEV_FAIL_OPEN:
    logger.error(
        "sensor_signing: 'cryptography' library not installed and "
        "HELI_SECURITY_DEV_FAIL_OPEN is not set. sign() and verify() will "
        "raise. Install cryptography (pip install cryptography) for production "
        "use. To skip in dev only, set HELI_SECURITY_DEV_FAIL_OPEN=1."
    )
elif not _CRYPTO_AVAILABLE and _DEV_FAIL_OPEN:
    logger.warning(
        "sensor_signing: cryptography NOT installed but "
        "HELI_SECURITY_DEV_FAIL_OPEN=1. Signatures will be empty and "
        "verification fail-open. NEVER ENABLE IN PRODUCTION."
    )


class SensorSigningUnavailable(RuntimeError):
    """Raised when crypto operations are attempted but the cryptography
    library is not installed and dev fail-open is not enabled."""


def _keys_dir(keys_dir: str = None) -> Path:
    base = keys_dir or os.environ.get("SUMMIT_SENSOR_KEYS_DIR", _DEFAULT_KEYS_DIR)
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Key generation & loading
# ---------------------------------------------------------------------------


def generate_keypair(
    sensor_id: str, keys_dir: str = None
) -> Tuple[bytes, bytes]:
    """
    Generate an Ed25519 keypair for *sensor_id*.  Saves raw seed (.key) and
    raw public key (.pub) to *keys_dir*.  Returns (private_seed_bytes, pubkey_bytes).

    Raises SensorSigningUnavailable if the cryptography library is not
    installed (unless HELI_SECURITY_DEV_FAIL_OPEN=1, in which case returns
    empty bytes — for dev only).
    """
    if not _CRYPTO_AVAILABLE:
        if _DEV_FAIL_OPEN:
            logger.warning("generate_keypair: dev fail-open mode — returning empty bytes")
            return (b"", b"")
        raise SensorSigningUnavailable(
            "generate_keypair requires cryptography library. "
            "Install it: pip install cryptography")

    d = _keys_dir(keys_dir)
    priv_path = d / f"{sensor_id}.key"
    pub_path = d / f"{sensor_id}.pub"

    private_key = Ed25519PrivateKey.generate()
    priv_bytes = private_key.private_bytes(
        Encoding.Raw, PrivateFormat.Raw, NoEncryption()
    )
    pub_bytes = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)

    priv_path.write_bytes(priv_bytes)
    pub_path.write_bytes(pub_bytes)
    priv_path.chmod(0o600)

    logger.info("generate_keypair: created keypair for sensor_id=%s at %s", sensor_id, d)
    return (priv_bytes, pub_bytes)


def load_private_key(sensor_id: str, keys_dir: str = None) -> Any:
    """Load and return an Ed25519PrivateKey for *sensor_id*."""
    if not _CRYPTO_AVAILABLE:
        return None
    priv_path = _keys_dir(keys_dir) / f"{sensor_id}.key"
    seed = priv_path.read_bytes()
    return Ed25519PrivateKey.from_private_bytes(seed)


def load_public_key(sensor_id: str, keys_dir: str = None) -> Any:
    """Load and return an Ed25519PublicKey for *sensor_id*."""
    if not _CRYPTO_AVAILABLE:
        return None
    pub_path = _keys_dir(keys_dir) / f"{sensor_id}.pub"
    raw = pub_path.read_bytes()
    return Ed25519PublicKey.from_public_bytes(raw)


# ---------------------------------------------------------------------------
# Sign / Verify
# ---------------------------------------------------------------------------


def sign_frame(sensor_id: str, payload: bytes, keys_dir: str = None) -> str:
    """
    Sign *payload* with the sensor's private key.
    Returns a base64url-encoded signature string.

    Raises SensorSigningUnavailable if cryptography is not installed and
    HELI_SECURITY_DEV_FAIL_OPEN is not set.
    """
    if not _CRYPTO_AVAILABLE:
        if _DEV_FAIL_OPEN:
            return ""
        raise SensorSigningUnavailable(
            "sign_frame requires cryptography library. "
            "Install it: pip install cryptography")

    private_key = load_private_key(sensor_id, keys_dir)
    sig_bytes = private_key.sign(payload)
    return base64.urlsafe_b64encode(sig_bytes).decode("ascii")


def verify_frame(
    sensor_id: str, payload: bytes, signature: str, keys_dir: str = None
) -> bool:
    """
    Verify *signature* over *payload* using the sensor's public key.
    Returns False on any error (bad signature, missing key file, malformed data).

    Fails CLOSED: returns False when cryptography is not installed (rather
    than the previous fail-OPEN True). Set HELI_SECURITY_DEV_FAIL_OPEN=1
    to revert to fail-open in dev environments only.
    """
    if not _CRYPTO_AVAILABLE:
        if _DEV_FAIL_OPEN:
            return True
        # Fail closed in production
        return False

    try:
        public_key = load_public_key(sensor_id, keys_dir)
        sig_bytes = base64.urlsafe_b64decode(signature + "==")
        public_key.verify(sig_bytes, payload)
        return True
    except InvalidSignature:
        logger.warning("verify_frame: invalid signature for sensor_id=%s", sensor_id)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("verify_frame: error for sensor_id=%s: %s", sensor_id, exc)
        return False
