"""
Heli.OS MFA — TOTP + WebAuthn + Backup Codes

TOTP secrets are AES-256-GCM encrypted before storage.
Backup codes are hashed with Argon2id.
WebAuthn follows the W3C Web Authentication spec.

Encryption scheme for TOTP secrets:
  - 12-byte random nonce (GCM standard)
  - AES-256-GCM authenticated encryption (256-bit key required)
  - Ciphertext, nonce, and GCM tag packed as base64-encoded JSON

WebAuthn is provided by the optional ``webauthn`` package. All WebAuthn
functions raise ImportError with installation instructions if the package
is not present.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import string
from typing import Optional

import pyotp
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .passwords import hash_password, verify_password

logger = logging.getLogger("security.mfa")

# ---------------------------------------------------------------------------
# WebAuthn optional import
# ---------------------------------------------------------------------------

try:
    import webauthn as _webauthn  # type: ignore
    from webauthn.helpers.structs import (  # type: ignore
        AuthenticatorSelectionCriteria,
        UserVerificationRequirement,
        ResidentKeyRequirement,
        AuthenticatorAttachment,
        PublicKeyCredentialDescriptor,
    )
    from webauthn.helpers.cose import COSEAlgorithmIdentifier  # type: ignore

    _WEBAUTHN_AVAILABLE = True
except ImportError:
    _WEBAUTHN_AVAILABLE = False

_WEBAUTHN_INSTALL_MSG = "pip install webauthn"


# ---------------------------------------------------------------------------
# TOTP — Secret Generation & Encryption
# ---------------------------------------------------------------------------


def generate_totp_secret() -> str:
    """
    Generate a cryptographically random base32 TOTP secret.

    The secret is 20 bytes (160 bits) of random data encoded as base32,
    which is the standard length used by Google Authenticator, Authy, and
    other RFC 6238 implementations.

    Returns:
        A base32-encoded TOTP secret string (uppercase, no padding).
    """
    return pyotp.random_base32()


def encrypt_secret(secret: str, key: bytes) -> str:
    """
    AES-256-GCM encrypt a TOTP secret for database storage.

    A fresh 12-byte nonce is generated for every encryption call.
    The nonce, ciphertext, and GCM authentication tag are bundled
    together in a base64-encoded JSON envelope so the format is
    self-describing and easy to extend.

    Args:
        secret: The plaintext base32 TOTP secret.
        key: A 32-byte (256-bit) AES key. Raise ValueError if not 32 bytes.

    Returns:
        A base64-encoded JSON string containing ``nonce``, ``ciphertext``,
        and ``tag`` fields (all individually base64-encoded).

    Raises:
        ValueError: If ``key`` is not exactly 32 bytes.
    """
    if len(key) != 32:
        raise ValueError(f"AES-256 key must be exactly 32 bytes, got {len(key)}")

    aesgcm = AESGCM(key)
    nonce = os.urandom(12)  # GCM standard nonce size
    plaintext = secret.encode("utf-8")

    # AESGCM.encrypt returns ciphertext + 16-byte GCM tag concatenated.
    ciphertext_with_tag = aesgcm.encrypt(nonce, plaintext, None)
    # Split: last 16 bytes are the tag.
    ciphertext = ciphertext_with_tag[:-16]
    tag = ciphertext_with_tag[-16:]

    envelope = {
        "v": 1,
        "alg": "AES-256-GCM",
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        "tag": base64.b64encode(tag).decode("ascii"),
    }
    raw = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def decrypt_secret(encrypted: str, key: bytes) -> str:
    """
    Decrypt an AES-256-GCM encrypted TOTP secret.

    Args:
        encrypted: The base64-encoded JSON envelope produced by
                   :func:`encrypt_secret`.
        key: The same 32-byte AES key used for encryption.

    Returns:
        The plaintext base32 TOTP secret.

    Raises:
        ValueError: If the key length is wrong, the envelope is malformed,
                    or GCM authentication fails (tampered ciphertext).
    """
    if len(key) != 32:
        raise ValueError(f"AES-256 key must be exactly 32 bytes, got {len(key)}")

    try:
        raw = base64.b64decode(encrypted)
        envelope = json.loads(raw)
        nonce = base64.b64decode(envelope["nonce"])
        ciphertext = base64.b64decode(envelope["ciphertext"])
        tag = base64.b64decode(envelope["tag"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed encrypted secret envelope: {exc}") from exc

    aesgcm = AESGCM(key)
    # cryptography library expects ciphertext + tag concatenated.
    ciphertext_with_tag = ciphertext + tag
    try:
        plaintext = aesgcm.decrypt(nonce, ciphertext_with_tag, None)
    except Exception as exc:
        raise ValueError("Decryption failed — ciphertext may be tampered") from exc

    return plaintext.decode("utf-8")


# ---------------------------------------------------------------------------
# TOTP — URI & QR Code
# ---------------------------------------------------------------------------


def get_totp_uri(secret: str, user_email: str, issuer: str = "Heli.OS") -> str:
    """
    Build a TOTP provisioning URI (otpauth://) suitable for QR codes.

    The URI format follows the Google Authenticator Key URI Format spec
    and is compatible with Authy, 1Password, Bitwarden, and other TOTP apps.

    Args:
        secret: The plaintext base32 TOTP secret.
        user_email: The user's email address (used as the account name).
        issuer: Human-readable service name shown in the authenticator app.

    Returns:
        An ``otpauth://totp/...`` URI string.
    """
    totp = pyotp.TOTP(secret)
    return totp.provisioning_uri(name=user_email, issuer_name=issuer)


def generate_qr_code_data_uri(totp_uri: str) -> str:
    """
    Render a TOTP provisioning URI as an inline PNG data URI.

    The returned string can be used directly as the ``src`` attribute of
    an HTML ``<img>`` tag, avoiding the need to serve the QR code image
    as a separate HTTP resource.

    Args:
        totp_uri: An ``otpauth://totp/...`` URI string.

    Returns:
        A ``data:image/png;base64,...`` data URI string.

    Raises:
        ImportError: If the ``qrcode[pil]`` package is not installed.
    """
    try:
        import qrcode  # type: ignore
        import qrcode.image.pil  # type: ignore
        from io import BytesIO
    except ImportError as exc:
        raise ImportError("pip install qrcode[pil]") from exc

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(totp_uri)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# ---------------------------------------------------------------------------
# TOTP — Verification
# ---------------------------------------------------------------------------


def verify_totp(token: str, secret: str, window: int = 1) -> bool:
    """
    Verify a 6-digit TOTP code against a secret.

    The ``window`` parameter allows for clock drift: a value of 1 accepts
    codes from ±1 time step (±30 seconds). This matches RFC 6238
    recommendations and what most authenticator apps expect.

    Args:
        token: The 6-digit code entered by the user.
        secret: The plaintext base32 TOTP secret.
        window: Number of time steps (30s each) to allow before/after the
                current time. Default 1 (±30s drift tolerance).

    Returns:
        True if the code is valid within the allowed window, False otherwise.
    """
    totp = pyotp.TOTP(secret)
    # pyotp's valid_window parameter counts steps on *each* side.
    return totp.verify(token, valid_window=window)


# ---------------------------------------------------------------------------
# Backup Codes
# ---------------------------------------------------------------------------

_BACKUP_CODE_ALPHABET = string.ascii_uppercase + string.digits


def generate_backup_codes(n: int = 10) -> list[str]:
    """
    Generate a list of one-time backup codes for MFA recovery.

    Codes are 8 characters, drawn from uppercase letters and digits
    (excluding visually ambiguous characters is a UX nicety; we keep the
    full set here since codes are copy-pasted, not hand-typed).

    Args:
        n: Number of backup codes to generate (default 10).

    Returns:
        A list of ``n`` random 8-character alphanumeric strings.
    """
    return [
        "".join(secrets.choice(_BACKUP_CODE_ALPHABET) for _ in range(8))
        for _ in range(n)
    ]


def hash_backup_code(code: str) -> str:
    """
    Hash a backup code using Argon2id for safe storage.

    Uses the same Argon2id parameters as password hashing (OWASP minimums).
    Backup codes are short (8 chars, ~47 bits) so the memory-hard hashing
    is especially important to prevent offline brute-force if the hash
    database is compromised.

    Args:
        code: The plaintext 8-character backup code.

    Returns:
        An Argon2id hash string safe for database storage.
    """
    return hash_password(code)


def verify_backup_code(code: str, hashes: list[str]) -> tuple[bool, Optional[str]]:
    """
    Check a submitted backup code against a list of stored Argon2id hashes.

    Iterates through all stored hashes and performs a constant-time
    comparison for each. Returns the matched hash so the caller can remove
    it from the store (backup codes are single-use).

    Args:
        code: The plaintext backup code submitted by the user.
        hashes: A list of Argon2id hash strings previously produced by
                :func:`hash_backup_code`.

    Returns:
        A ``(valid, matched_hash)`` tuple. ``matched_hash`` is the hash
        string that verified successfully, or ``None`` if no match was found.
    """
    for stored_hash in hashes:
        if verify_password(code, stored_hash):
            return True, stored_hash
    return False, None


# ---------------------------------------------------------------------------
# WebAuthn / FIDO2
# ---------------------------------------------------------------------------


def _require_webauthn() -> None:
    """Raise ImportError if the webauthn package is unavailable."""
    if not _WEBAUTHN_AVAILABLE:
        raise ImportError(_WEBAUTHN_INSTALL_MSG)


def begin_registration(
    user_id: str,
    user_name: str,
    rp_id: str,
    rp_name: str,
    existing_credentials: list[dict] | None = None,
) -> dict:
    """
    Generate WebAuthn registration options for a new authenticator.

    Wraps ``webauthn.generate_registration_options()`` and returns a
    JSON-serialisable dict that the frontend passes to
    ``navigator.credentials.create()``.

    Args:
        user_id: Unique, opaque identifier for the user (not displayed).
        user_name: Human-readable account name (shown in authenticator UI).
        rp_id: Relying Party ID — the effective domain (e.g. ``example.com``).
        rp_name: Human-readable RP name shown in the security key prompt.
        existing_credentials: List of already-registered credential dicts
            (each must have a ``credential_id`` key) to exclude from the
            prompt, preventing duplicate registrations.

    Returns:
        A JSON-serialisable dict of registration options.

    Raises:
        ImportError: If the ``webauthn`` package is not installed.
    """
    _require_webauthn()

    exclude = []
    for cred in existing_credentials or []:
        exclude.append(
            PublicKeyCredentialDescriptor(id=bytes.fromhex(cred["credential_id"]))
        )

    options = _webauthn.generate_registration_options(
        rp_id=rp_id,
        rp_name=rp_name,
        user_id=user_id.encode("utf-8"),
        user_name=user_name,
        exclude_credentials=exclude,
        authenticator_selection=AuthenticatorSelectionCriteria(
            resident_key=ResidentKeyRequirement.PREFERRED,
            user_verification=UserVerificationRequirement.PREFERRED,
        ),
        supported_pub_key_algs=[
            COSEAlgorithmIdentifier.ECDSA_SHA_256,
            COSEAlgorithmIdentifier.RSASSA_PKCS1_v1_5_SHA_256,
        ],
    )

    return json.loads(_webauthn.options_to_json(options))


def complete_registration(
    credential: dict,
    challenge: str,
    rp_id: str,
    expected_origin: str,
) -> dict:
    """
    Verify a WebAuthn registration response and extract credential data.

    Args:
        credential: The JSON-decoded credential object returned by
                    ``navigator.credentials.create()``.
        challenge: The base64url-encoded challenge that was sent to the client
                   (from :func:`begin_registration` response).
        rp_id: The Relying Party ID used during registration.
        expected_origin: The full origin the browser used (e.g.
                         ``https://example.com``).

    Returns:
        A dict with keys: ``credential_id`` (hex str), ``public_key`` (hex
        str), ``sign_count`` (int), ``aaguid`` (str), ``device_name`` (str).

    Raises:
        ImportError: If the ``webauthn`` package is not installed.
        Exception: If registration verification fails for any reason.
    """
    _require_webauthn()

    from webauthn.helpers.structs import RegistrationCredential  # type: ignore

    reg_credential = RegistrationCredential.parse_raw(json.dumps(credential))
    challenge_bytes = base64.urlsafe_b64decode(
        challenge + "=" * (4 - len(challenge) % 4)
    )

    verified = _webauthn.verify_registration_response(
        credential=reg_credential,
        expected_challenge=challenge_bytes,
        expected_rp_id=rp_id,
        expected_origin=expected_origin,
        require_user_verification=False,
    )

    aaguid = str(verified.aaguid) if verified.aaguid else ""
    return {
        "credential_id": verified.credential_id.hex(),
        "public_key": verified.credential_public_key.hex(),
        "sign_count": verified.sign_count,
        "aaguid": aaguid,
        "device_name": _guess_device_name(aaguid),
    }


def begin_authentication(
    credentials: list[dict],
    rp_id: str,
) -> dict:
    """
    Generate WebAuthn authentication options for a login attempt.

    Args:
        credentials: List of stored credential dicts for the user. Each must
                     have a ``credential_id`` (hex str) key.
        rp_id: The Relying Party ID.

    Returns:
        A JSON-serialisable dict of authentication options to send to the
        browser's ``navigator.credentials.get()``.

    Raises:
        ImportError: If the ``webauthn`` package is not installed.
    """
    _require_webauthn()

    allow = [
        PublicKeyCredentialDescriptor(id=bytes.fromhex(c["credential_id"]))
        for c in credentials
    ]

    options = _webauthn.generate_authentication_options(
        rp_id=rp_id,
        allow_credentials=allow,
        user_verification=UserVerificationRequirement.PREFERRED,
    )

    return json.loads(_webauthn.options_to_json(options))


def complete_authentication(
    credential: dict,
    challenge: str,
    stored_credential: dict,
    rp_id: str,
    expected_origin: str,
) -> dict:
    """
    Verify a WebAuthn authentication assertion.

    Args:
        credential: The JSON-decoded credential returned by
                    ``navigator.credentials.get()``.
        challenge: The base64url-encoded challenge from
                   :func:`begin_authentication`.
        stored_credential: The stored credential record for this user
                           (must have ``credential_id``, ``public_key``,
                           and ``sign_count`` keys).
        rp_id: The Relying Party ID.
        expected_origin: The full origin the browser used.

    Returns:
        A dict with key ``sign_count`` (int) — the new counter value that
        must be persisted to prevent replay attacks.

    Raises:
        ImportError: If the ``webauthn`` package is not installed.
        Exception: If authentication verification fails.
    """
    _require_webauthn()

    from webauthn.helpers.structs import AuthenticationCredential  # type: ignore

    auth_credential = AuthenticationCredential.parse_raw(json.dumps(credential))
    challenge_bytes = base64.urlsafe_b64decode(
        challenge + "=" * (4 - len(challenge) % 4)
    )
    public_key_bytes = bytes.fromhex(stored_credential["public_key"])

    verified = _webauthn.verify_authentication_response(
        credential=auth_credential,
        expected_challenge=challenge_bytes,
        expected_rp_id=rp_id,
        expected_origin=expected_origin,
        credential_public_key=public_key_bytes,
        credential_current_sign_count=stored_credential["sign_count"],
        require_user_verification=False,
    )

    return {"sign_count": verified.new_sign_count}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _guess_device_name(aaguid: str) -> str:
    """
    Return a human-readable device name for well-known AAGUIDs.

    Falls back to "Security Key" for unknown authenticators.
    YubiKey AAGUIDs sourced from Yubico's official metadata and the
    FIDO Alliance Metadata Service (MDS3).
    """
    _KNOWN: dict[str, str] = {
        # ── YubiKey 5 Series ────────────────────────────────────────────────
        "2fc0579f-8113-47ea-b116-bb5a8db9202a": "YubiKey 5 NFC",
        "6d44ba9b-f6ec-2e49-b930-0c8fe920cb73": "YubiKey 5Ci",
        "cb69481e-8ff7-4039-93ec-0a2729a154a8": "YubiKey 5 USB-C",
        "c5ef55ff-ad9a-4b9f-b580-adebafe026d0": "YubiKey 5Ci FIPS",
        "c1f9a0bc-1dd2-404a-b27f-8e29047a43fd": "YubiKey 5 NFC FIPS",
        "73bb0cd4-e502-49b8-9c6f-b59445bf720b": "YubiKey 5C NFC FIPS",
        "85203421-48f9-4355-9bc8-8a53846e5083": "YubiKey 5 USB-C FIPS",
        "a4e9fc6d-4cbe-4758-b8ba-37598bb5bbaa": "YubiKey 5C USB",
        "ee882879-721c-4913-9775-3dfcce97072a": "YubiKey 5 NFC (USB-A)",
        "fa2b99dc-9e39-4257-8f92-4a30d23c4118": "YubiKey 5 Series",
        "0bb43545-fd2c-4185-87dd-feb0b2916ace": "YubiKey 5 NFC Enterprise",
        # ── YubiKey Bio Series ───────────────────────────────────────────────
        "d8522d9f-575b-4866-88a9-ba99fa02f35b": "YubiKey Bio Edition",
        "149a2021-8ef6-4133-96b8-81f8d5b7f1f5": "YubiKey Bio Multi-Protocol Edition",
        "7359616b-09b7-44e0-a980-c2b9a7a8e9cd": "YubiKey Bio USB-C",
        # ── YubiKey Security Keys ────────────────────────────────────────────
        "f8a011f3-8c0a-4d15-8006-17111f9edc7d": "Security Key NFC by Yubico",
        "b92c3f9a-c014-4056-887f-140a2501163b": "Security Key NFC by Yubico Enterprise",
        # ── YubiKey 4 / NEO (legacy) ─────────────────────────────────────────
        "116cc006-28d4-4927-9d3f-2b9d3e84d3e4": "YubiKey 4/4 Nano",
        # ── Google Titan ──────────────────────────────────────────────────────
        "42b4fb4a-2866-43b2-9bf7-6c6669c2e5d3": "Google Titan Security Key",
        "b67e0a21-3ce7-4a08-9e35-73ee7f1fe6b5": "Google Titan Security Key USB-C",
        # ── Feitian ───────────────────────────────────────────────────────────
        "12ded745-4bed-47d4-abaa-e713f51d6393": "Feitian ePass FIDO2",
        "9c835346-796b-4c27-8898-d6032f515cc5": "Feitian BioPass FIDO2",
        "77010bd7-212a-4fc9-b236-d2ca5e9d4084": "Feitian MultiPass FIDO2",
        # ── SoloKeys ─────────────────────────────────────────────────────────
        "8876631b-d4a0-427f-5773-0ec71c9e0279": "Solo 1",
        "370b5bc0-e460-44ba-9c4a-c93a7df41f4e": "Solo Tap",
        # ── Platform authenticators ───────────────────────────────────────────
        "00000000-0000-0000-0000-000000000000": "Virtual Authenticator",
        "adce0002-35bc-c60a-648b-0b25f1f05503": "Chrome on macOS (Touch ID)",
        "08987058-cadc-4b81-b6e1-30de50dcbe96": "Windows Hello",
        "9ddd1817-af5a-4672-a2b9-3e3dd95000a9": "Windows Hello",
        "6028b017-b1d4-4c02-b4b3-afcdafc96bb2": "Windows Hello",
        "dd4ec289-e01d-41c9-bb89-70fa845d4bf2": "iCloud Keychain",
        "531126d6-e717-415c-9320-3d9aa6981239": "Dashlane",
        "b5397666-4885-aa6b-cebf-e52262a439a2": "Chromium Browser",
        "ea9b8d66-4d01-1d21-3ce4-b6b48cb575d4": "Google Password Manager",
    }
    return _KNOWN.get(aaguid, "Security Key")


# ---------------------------------------------------------------------------
# YubiKey OTP — Yubico OTP protocol (distinct from FIDO2/WebAuthn)
# ---------------------------------------------------------------------------
# YubiKey OTP generates a 44-character one-time password (modhex-encoded).
# Validated via Yubico's YubiCloud API (free) or a self-hosted KSM.
#
# Setup:
#   1. Get an API key at https://upgrade.yubico.com/getapikey/
#   2. Set YUBICO_CLIENT_ID and YUBICO_SECRET_KEY in your environment.
#   3. At registration, store get_yubikey_identity(otp) per user.
#   4. At login, call verify_yubikey_otp(otp, expected_identity=stored_id).


def get_yubikey_identity(otp: str) -> str:
    """
    Extract the 12-character public identity prefix from a YubiKey OTP.

    The identity is fixed for a given physical key. Store it per-user at
    registration to bind the key to an account and reject cross-user tokens.

    Args:
        otp: A 44-character Yubico OTP string.

    Returns:
        The first 12 characters of the OTP (the key's public identity).

    Raises:
        ValueError: If the OTP is not exactly 44 characters.
    """
    otp = otp.strip().lower()
    if len(otp) != 44:
        raise ValueError(f"YubiKey OTP must be 44 characters, got {len(otp)}")
    return otp[:12]


async def verify_yubikey_otp(
    otp: str,
    expected_identity: Optional[str] = None,
    client_id: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> dict:
    """
    Verify a Yubico OTP against the YubiCloud validation API.

    Args:
        otp: The 44-character OTP generated by the YubiKey.
        expected_identity: If provided, the first 12 chars of the OTP must
                           match (prevents cross-account token replay).
        client_id: Yubico API client ID. Falls back to YUBICO_CLIENT_ID env var.
        secret_key: Yubico API HMAC-SHA1 secret. Falls back to YUBICO_SECRET_KEY.

    Returns:
        Dict with:
          ``valid`` (bool)  — True if the OTP is genuine and not replayed.
          ``identity`` (str) — The 12-char public key identity.
          ``status`` (str)  — Raw YubiCloud status (OK, REPLAYED_OTP, BAD_OTP, …).

    Raises:
        ValueError: If OTP length is wrong or identity doesn't match.
        ImportError: If yubico-client is not installed (pip install yubico-client).
        RuntimeError: If YUBICO_CLIENT_ID is not configured.
    """
    try:
        import yubico_client  # type: ignore
        import yubico_client.yubico_exceptions  # type: ignore
    except ImportError:
        raise ImportError(
            "yubico-client is not installed. Run: pip install yubico-client"
        )

    otp = otp.strip().lower()
    if len(otp) != 44:
        raise ValueError(f"YubiKey OTP must be 44 characters, got {len(otp)}")

    identity = otp[:12]
    if expected_identity and identity != expected_identity.lower():
        raise ValueError(
            f"OTP identity mismatch: expected {expected_identity}, got {identity}. "
            "This hardware key is not registered to this account."
        )

    _client_id = client_id or os.getenv("YUBICO_CLIENT_ID", "")
    _secret = secret_key or os.getenv("YUBICO_SECRET_KEY", "")

    if not _client_id:
        raise RuntimeError(
            "YUBICO_CLIENT_ID is not set. "
            "Get a free API key at https://upgrade.yubico.com/getapikey/ "
            "and set YUBICO_CLIENT_ID and YUBICO_SECRET_KEY in your environment."
        )

    try:
        client = yubico_client.Yubico(
            client_id=_client_id,
            key=_secret or None,
        )
        import asyncio as _asyncio
        result = await _asyncio.to_thread(client.verify, otp)
        return {"valid": bool(result), "identity": identity, "status": "OK"}
    except yubico_client.yubico_exceptions.StatusCodeError as exc:
        status = str(exc)
        if "REPLAYED_OTP" in status:
            return {"valid": False, "identity": identity, "status": "REPLAYED_OTP"}
        if "BAD_OTP" in status:
            return {"valid": False, "identity": identity, "status": "BAD_OTP"}
        return {"valid": False, "identity": identity, "status": status}
    except Exception as exc:
        logger.error("YubiCloud verification error: %s", exc)
        return {"valid": False, "identity": identity, "status": f"ERROR: {exc}"}
