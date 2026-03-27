"""
packages/security/field_encryption.py — AES-256-GCM field-level encryption.

Encrypts individual database column values (PII: org names, emails) at the
application layer so they're opaque to DB-level access.

Key configuration:
  FIELD_ENCRYPTION_KEY — base64-encoded 32-byte key (use: openssl rand -base64 32)

If the key is absent, plaintext is stored with a WARNING logged once. This
keeps the service operational in dev/test environments without a key.

Wire-format for encrypted values:
  Base64URL( version_byte || nonce[12] || ciphertext || tag[16] )
  version_byte = 0x01
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Optional

logger = logging.getLogger("security.field_encryption")

_VERSION = b"\x01"
_KEY: Optional[bytes] = None
_warned = False


def _get_key() -> Optional[bytes]:
    """Lazily load and cache the encryption key."""
    global _KEY, _warned
    if _KEY is not None:
        return _KEY

    raw = os.getenv("FIELD_ENCRYPTION_KEY", "")
    if not raw:
        if not _warned:
            logger.warning(
                "FIELD_ENCRYPTION_KEY not set — PII fields stored as plaintext. "
                "Set FIELD_ENCRYPTION_KEY=<base64 32-byte key> in production."
            )
            _warned = True
        return None

    try:
        key = base64.b64decode(raw)
        if len(key) != 32:
            raise ValueError(f"key must be 32 bytes, got {len(key)}")
        _KEY = key
        logger.info("Field encryption key loaded (%d bytes)", len(key))
        return _KEY
    except Exception as exc:
        logger.error("FIELD_ENCRYPTION_KEY decode failed: %s", exc)
        return None


def encrypt_field(plaintext: str) -> str:
    """
    Encrypt a string field.

    Returns the encrypted value (versioned base64url string) ready for DB storage.
    If no key is configured, returns plaintext unchanged.
    """
    key = _get_key()
    if key is None or not plaintext:
        return plaintext

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        import os as _os

        nonce = _os.urandom(12)
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        payload = _VERSION + nonce + ciphertext
        return base64.urlsafe_b64encode(payload).decode("ascii")
    except Exception as exc:
        logger.error("Field encryption failed (returning plaintext): %s", exc)
        return plaintext


def decrypt_field(stored: str) -> str:
    """
    Decrypt a field encrypted by encrypt_field().

    If stored value is plaintext (no key was set when it was written, or key
    absent now), returns it unchanged. Safe to call on already-plaintext data.
    """
    if not stored:
        return stored

    # Detect our wire format: base64url, starts with version byte 0x01 after decode
    try:
        raw = base64.urlsafe_b64decode(stored + "==")  # pad for safety
        if raw[:1] != _VERSION:
            return stored  # plaintext or unknown format
    except Exception:
        return stored  # not base64 — must be plaintext

    key = _get_key()
    if key is None:
        logger.warning(
            "Cannot decrypt field — FIELD_ENCRYPTION_KEY not set; returning ciphertext as-is"
        )
        return stored

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        nonce = raw[1:13]
        ciphertext = raw[13:]
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ciphertext, None).decode("utf-8")
    except Exception as exc:
        logger.error("Field decryption failed: %s", exc)
        return stored
