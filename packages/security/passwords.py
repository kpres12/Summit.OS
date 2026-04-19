"""
Argon2id password hashing for Heli.OS.

Production parameters: time_cost=3, memory_cost=65536 (64MB), parallelism=4
These are OWASP recommended minimums for Argon2id.

The full Argon2id hash string is self-describing and includes the algorithm,
version, parameters, salt, and digest — making it safe to store directly in
the database and enabling parameter upgrades via needs_rehash().
"""

from __future__ import annotations

import secrets
import base64
import logging

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError

logger = logging.getLogger("security.passwords")

# OWASP-recommended Argon2id parameters (2024).
# memory_cost is in KiB: 65536 KiB = 64 MiB.
_HASHER = PasswordHasher(
    time_cost=3,
    memory_cost=65536,
    parallelism=4,
    hash_len=32,
    salt_len=16,
)


def hash_password(password: str) -> str:
    """
    Hash a plaintext password using Argon2id.

    Returns the full encoded Argon2id hash string, which includes the
    algorithm identifier, version, parameters, salt, and digest. This
    string is safe to store directly in the database.

    Args:
        password: The plaintext password to hash.

    Returns:
        A self-describing Argon2id hash string of the form:
        ``$argon2id$v=19$m=65536,t=3,p=4$<salt>$<hash>``

    Raises:
        argon2.exceptions.HashingError: If hashing fails for any reason.
    """
    return _HASHER.hash(password)


def verify_password(password: str, hash: str) -> bool:
    """
    Verify a plaintext password against a stored Argon2id hash.

    Uses constant-time comparison internally (provided by argon2-cffi /
    libargon2). Returns False for any invalid input rather than raising,
    so callers do not need to distinguish between wrong-password and
    malformed-hash at the authentication decision point.

    Args:
        password: The plaintext password to check.
        hash: The stored Argon2id hash string.

    Returns:
        True if the password matches the hash, False otherwise.
    """
    try:
        return _HASHER.verify(hash, password)
    except VerifyMismatchError:
        # Password does not match — not an error condition, just False.
        return False
    except (VerificationError, InvalidHashError) as exc:
        logger.warning("Password verification error (malformed hash?): %s", exc)
        return False


def needs_rehash(hash: str) -> bool:
    """
    Return True if the stored hash was created with outdated parameters.

    Call this after a successful verify_password(). If it returns True,
    rehash the plaintext password (which you have at login time) and store
    the new hash so the account is silently upgraded to current parameters.

    Args:
        hash: The stored Argon2id hash string.

    Returns:
        True if the hash should be recomputed with current parameters.
    """
    return _HASHER.check_needs_rehash(hash)


def generate_secure_token(n_bytes: int = 32) -> str:
    """
    Generate a cryptographically secure, URL-safe random token.

    Uses os.urandom() via the ``secrets`` module. The token is base64url-
    encoded without padding, making it safe for use in URLs, cookies, and
    Authorization headers.

    Args:
        n_bytes: Number of random bytes to generate (default 32 → 256 bits).
                 The returned string will be longer due to base64 expansion.

    Returns:
        A URL-safe base64-encoded string with no padding characters.
    """
    raw = secrets.token_bytes(n_bytes)
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
