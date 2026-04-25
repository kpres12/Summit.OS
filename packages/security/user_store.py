"""
User MFA state store for Heli.OS.

Stores per-user MFA enrollment state:
- TOTP secrets (AES-256-GCM encrypted)
- WebAuthn credentials (public keys, sign counts)
- Backup code hashes
- Active sessions
- Login attempt rate limiting data

Schema is forward-compatible with Postgres via SQLAlchemy Core.

For development, pass a SQLite URL:
    sqlite+aiosqlite:///./summit_mfa.db
    sqlite+aiosqlite://   (in-memory)

For production, pass a PostgreSQL URL:
    postgresql+asyncpg://user:pass@host:5432/dbname

All public methods are async and safe to call from FastAPI route handlers.
"""

from __future__ import annotations

import json
import logging
import secrets
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    select,
    update,
    insert,
    func,
)
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from .mfa import encrypt_secret, decrypt_secret, verify_backup_code

logger = logging.getLogger("security.user_store")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_metadata = MetaData()

user_mfa = Table(
    "user_mfa",
    _metadata,
    Column("user_id", String(128), primary_key=True),
    Column("email", Text, nullable=False),
    Column("totp_secret_enc", Text, nullable=True),  # AES-256-GCM encrypted base64
    Column("totp_enabled", Integer, default=0),
    Column("totp_enrolled_at", DateTime(timezone=True), nullable=True),
    Column("backup_codes_json", Text, nullable=True),  # JSON array of Argon2id hashes
    Column("backup_codes_remaining", Integer, default=0),
    Column(
        "mfa_method", String(32), default="none"
    ),  # 'none', 'totp', 'webauthn', 'both', 'yubikey_otp'
    Column("yubikey_identity", String(12), nullable=True),  # 12-char Yubico OTP public id
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

webauthn_credentials = Table(
    "webauthn_credentials",
    _metadata,
    Column("credential_id", String(512), primary_key=True),
    Column("user_id", String(128), nullable=False),
    Column("public_key", Text, nullable=False),
    Column("sign_count", Integer, default=0),
    Column("aaguid", String(64), nullable=True),
    Column("device_name", String(256), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("last_used_at", DateTime(timezone=True), nullable=True),
    Column("active", Integer, default=1),
)

user_sessions = Table(
    "user_sessions",
    _metadata,
    Column("session_id", String(128), primary_key=True),
    Column("user_id", String(128), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Column("ip_address", String(64), nullable=True),
    Column("user_agent", Text, nullable=True),
    Column("mfa_verified", Integer, default=0),
    Column("revoked", Integer, default=0),
)

login_attempts = Table(
    "login_attempts",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("identifier", String(256), nullable=False),  # email or IP
    Column("success", Integer, default=0),
    Column("attempted_at", DateTime(timezone=True), nullable=False),
    Column("ip_address", String(64), nullable=True),
)

# Index for rate-limiting queries.
_idx_login_attempts = Index(
    "idx_login_attempts",
    login_attempts.c.identifier,
    login_attempts.c.attempted_at,
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _row_to_dict(row: Any) -> dict:
    """Convert a SQLAlchemy Row to a plain dict."""
    return dict(row._mapping)


# ---------------------------------------------------------------------------
# UserMFAStore
# ---------------------------------------------------------------------------


class UserMFAStore:
    """
    Async SQLAlchemy-based store for per-user MFA state.

    All I/O operations are async. The store is safe to share across
    multiple FastAPI workers as long as the database supports concurrency
    (PostgreSQL does; SQLite in WAL mode works fine for development).

    Args:
        db_url: SQLAlchemy async database URL.
        mfa_encryption_key: 32-byte AES-256 key used to encrypt TOTP secrets
            at rest. Must be kept secret — rotate with a key-migration job if
            compromised.
    """

    def __init__(self, db_url: str, mfa_encryption_key: bytes) -> None:
        if len(mfa_encryption_key) != 32:
            raise ValueError(
                f"mfa_encryption_key must be 32 bytes, got {len(mfa_encryption_key)}"
            )
        self._engine: AsyncEngine = create_async_engine(db_url, echo=False, future=True)
        self._key = mfa_encryption_key

    async def init_db(self) -> None:
        """
        Create all MFA tables if they do not already exist.

        Safe to call on every application startup (idempotent).
        """
        async with self._engine.begin() as conn:
            await conn.run_sync(_metadata.create_all)
        logger.info("UserMFAStore: database tables initialised")

    # ------------------------------------------------------------------
    # User record
    # ------------------------------------------------------------------

    async def get_user(self, user_id: str) -> Optional[dict]:
        """
        Retrieve the MFA record for a user.

        Args:
            user_id: The user's unique identifier.

        Returns:
            A dict of all user_mfa columns, or None if the user has no
            MFA record yet.
        """
        async with self._engine.connect() as conn:
            result = await conn.execute(
                select(user_mfa).where(user_mfa.c.user_id == user_id)
            )
            row = result.fetchone()
        return _row_to_dict(row) if row else None

    async def upsert_user(self, user_id: str, email: str) -> dict:
        """
        Create a user MFA record if one doesn't exist, or return the existing one.

        Args:
            user_id: Unique user identifier.
            email: User's email address (stored for display purposes).

        Returns:
            The current user_mfa record as a dict.
        """
        existing = await self.get_user(user_id)
        if existing:
            return existing

        now = _now()
        record = {
            "user_id": user_id,
            "email": email,
            "totp_secret_enc": None,
            "totp_enabled": 0,
            "totp_enrolled_at": None,
            "backup_codes_json": None,
            "backup_codes_remaining": 0,
            "mfa_method": "none",
            "created_at": now,
            "updated_at": now,
        }
        async with self._engine.begin() as conn:
            await conn.execute(insert(user_mfa).values(**record))
        logger.info("UserMFAStore: created MFA record for user %s", user_id)
        return record

    # ------------------------------------------------------------------
    # TOTP secrets
    # ------------------------------------------------------------------

    async def set_totp_secret(self, user_id: str, secret: str) -> None:
        """
        Encrypt and store a TOTP secret for a user.

        The secret is encrypted with AES-256-GCM using the store's key
        before being written to the database. The plaintext secret never
        touches disk.

        Args:
            user_id: The user's unique identifier.
            secret: The plaintext base32 TOTP secret.
        """
        encrypted = encrypt_secret(secret, self._key)
        async with self._engine.begin() as conn:
            await conn.execute(
                update(user_mfa)
                .where(user_mfa.c.user_id == user_id)
                .values(totp_secret_enc=encrypted, updated_at=_now())
            )

    async def get_totp_secret(self, user_id: str) -> Optional[str]:
        """
        Retrieve and decrypt the TOTP secret for a user.

        Args:
            user_id: The user's unique identifier.

        Returns:
            The plaintext base32 TOTP secret, or None if not enrolled.
        """
        record = await self.get_user(user_id)
        if not record or not record.get("totp_secret_enc"):
            return None
        return decrypt_secret(record["totp_secret_enc"], self._key)

    async def enable_totp(self, user_id: str, backup_codes: list[str]) -> None:
        """
        Activate TOTP for a user and store their hashed backup codes.

        Backup codes are hashed with Argon2id before storage. Callers must
        return the plaintext codes to the user *before* calling this method,
        because they cannot be recovered afterward.

        Args:
            user_id: The user's unique identifier.
            backup_codes: List of plaintext backup codes (will be hashed).
        """
        from .mfa import hash_backup_code

        hashes = [hash_backup_code(code) for code in backup_codes]
        now = _now()

        # Determine current webauthn state to set mfa_method correctly.
        creds = await self.get_webauthn_credentials(user_id)
        method = "both" if creds else "totp"

        async with self._engine.begin() as conn:
            await conn.execute(
                update(user_mfa)
                .where(user_mfa.c.user_id == user_id)
                .values(
                    totp_enabled=1,
                    totp_enrolled_at=now,
                    backup_codes_json=json.dumps(hashes),
                    backup_codes_remaining=len(hashes),
                    mfa_method=method,
                    updated_at=now,
                )
            )
        logger.info("UserMFAStore: TOTP enabled for user %s", user_id)

    async def disable_totp(self, user_id: str) -> None:
        """
        Disable TOTP for a user and clear their stored secret and backup codes.

        This does not remove WebAuthn credentials. The mfa_method is updated
        to reflect the remaining enrollment state.

        Args:
            user_id: The user's unique identifier.
        """
        creds = await self.get_webauthn_credentials(user_id)
        method = "webauthn" if creds else "none"

        async with self._engine.begin() as conn:
            await conn.execute(
                update(user_mfa)
                .where(user_mfa.c.user_id == user_id)
                .values(
                    totp_secret_enc=None,
                    totp_enabled=0,
                    totp_enrolled_at=None,
                    backup_codes_json=None,
                    backup_codes_remaining=0,
                    mfa_method=method,
                    updated_at=_now(),
                )
            )
        logger.info("UserMFAStore: TOTP disabled for user %s", user_id)

    # ------------------------------------------------------------------
    # Backup codes
    # ------------------------------------------------------------------

    async def verify_backup_code(self, user_id: str, code: str) -> bool:
        """
        Verify a backup code and remove it from the user's stored set.

        Backup codes are single-use. If the submitted code matches any stored
        hash, that hash is removed and the remaining count is decremented.

        Args:
            user_id: The user's unique identifier.
            code: The plaintext backup code submitted by the user.

        Returns:
            True if the code was valid (and has now been consumed), False
            otherwise.
        """
        record = await self.get_user(user_id)
        if not record or not record.get("backup_codes_json"):
            return False

        stored_hashes: list[str] = json.loads(record["backup_codes_json"])
        valid, matched = verify_backup_code(code, stored_hashes)

        if not valid or matched is None:
            return False

        # Remove the used hash.
        remaining = [h for h in stored_hashes if h != matched]
        async with self._engine.begin() as conn:
            await conn.execute(
                update(user_mfa)
                .where(user_mfa.c.user_id == user_id)
                .values(
                    backup_codes_json=json.dumps(remaining),
                    backup_codes_remaining=len(remaining),
                    updated_at=_now(),
                )
            )
        logger.info(
            "UserMFAStore: backup code consumed for user %s (%d remaining)",
            user_id,
            len(remaining),
        )
        return True

    # ------------------------------------------------------------------
    # WebAuthn credentials
    # ------------------------------------------------------------------

    async def add_webauthn_credential(self, user_id: str, cred: dict) -> None:
        """
        Persist a newly registered WebAuthn credential.

        Args:
            user_id: The user's unique identifier.
            cred: Credential dict with keys: ``credential_id``, ``public_key``,
                  ``sign_count``, ``aaguid``, ``device_name``.
        """
        now = _now()
        async with self._engine.begin() as conn:
            await conn.execute(
                insert(webauthn_credentials).values(
                    credential_id=cred["credential_id"],
                    user_id=user_id,
                    public_key=cred["public_key"],
                    sign_count=cred.get("sign_count", 0),
                    aaguid=cred.get("aaguid", ""),
                    device_name=cred.get("device_name", "Security Key"),
                    created_at=now,
                    last_used_at=None,
                    active=1,
                )
            )
            # Update mfa_method on the user record.
            record = await conn.execute(
                select(user_mfa.c.totp_enabled).where(user_mfa.c.user_id == user_id)
            )
            row = record.fetchone()
            totp_on = bool(row[0]) if row else False
            method = "both" if totp_on else "webauthn"
            await conn.execute(
                update(user_mfa)
                .where(user_mfa.c.user_id == user_id)
                .values(mfa_method=method, updated_at=now)
            )
        logger.info(
            "UserMFAStore: WebAuthn credential %s added for user %s",
            cred["credential_id"],
            user_id,
        )

    async def get_webauthn_credentials(self, user_id: str) -> list[dict]:
        """
        Return all active WebAuthn credentials for a user.

        Args:
            user_id: The user's unique identifier.

        Returns:
            A list of credential dicts (may be empty).
        """
        async with self._engine.connect() as conn:
            result = await conn.execute(
                select(webauthn_credentials)
                .where(
                    webauthn_credentials.c.user_id == user_id,
                    webauthn_credentials.c.active == 1,
                )
                .order_by(webauthn_credentials.c.created_at)
            )
            rows = result.fetchall()
        return [_row_to_dict(r) for r in rows]

    async def update_webauthn_sign_count(
        self, credential_id: str, sign_count: int
    ) -> None:
        """
        Update the sign count for a WebAuthn credential after successful auth.

        Storing and checking the sign count is a key anti-replay mechanism
        specified in the WebAuthn spec. Always call this after a successful
        :func:`~security.mfa.complete_authentication`.

        Args:
            credential_id: The credential's unique identifier (hex string).
            sign_count: The new sign count returned by the authenticator.
        """
        async with self._engine.begin() as conn:
            await conn.execute(
                update(webauthn_credentials)
                .where(webauthn_credentials.c.credential_id == credential_id)
                .values(sign_count=sign_count, last_used_at=_now())
            )

    async def revoke_webauthn_credential(
        self, credential_id: str, user_id: str
    ) -> None:
        """
        Soft-delete a WebAuthn credential.

        The credential is marked inactive rather than deleted so the audit
        trail is preserved.

        Args:
            credential_id: The credential to revoke.
            user_id: Must match the credential's owner (prevents IDOR).
        """
        async with self._engine.begin() as conn:
            await conn.execute(
                update(webauthn_credentials)
                .where(
                    webauthn_credentials.c.credential_id == credential_id,
                    webauthn_credentials.c.user_id == user_id,
                )
                .values(active=0)
            )
            # Re-evaluate mfa_method.
            remaining = await conn.execute(
                select(func.count())
                .select_from(webauthn_credentials)
                .where(
                    webauthn_credentials.c.user_id == user_id,
                    webauthn_credentials.c.active == 1,
                )
            )
            count = remaining.scalar() or 0
            totp_row = await conn.execute(
                select(user_mfa.c.totp_enabled).where(user_mfa.c.user_id == user_id)
            )
            totp_on = bool((totp_row.fetchone() or (0,))[0])
            if count > 0 and totp_on:
                method = "both"
            elif count > 0:
                method = "webauthn"
            elif totp_on:
                method = "totp"
            else:
                method = "none"
            await conn.execute(
                update(user_mfa)
                .where(user_mfa.c.user_id == user_id)
                .values(mfa_method=method, updated_at=_now())
            )
        logger.info(
            "UserMFAStore: credential %s revoked for user %s", credential_id, user_id
        )

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    async def create_session(
        self,
        user_id: str,
        expires_in: int,
        ip: str,
        ua: str,
        mfa_verified: bool,
    ) -> str:
        """
        Create a new authenticated session.

        Args:
            user_id: The authenticated user's identifier.
            expires_in: Session lifetime in seconds from now.
            ip: The client's IP address.
            ua: The client's User-Agent string.
            mfa_verified: Whether MFA was completed for this session.

        Returns:
            The new session ID (256-bit URL-safe random token).
        """
        session_id = secrets.token_urlsafe(32)
        now = _now()
        expires_at = now + timedelta(seconds=expires_in)

        async with self._engine.begin() as conn:
            await conn.execute(
                insert(user_sessions).values(
                    session_id=session_id,
                    user_id=user_id,
                    created_at=now,
                    expires_at=expires_at,
                    ip_address=ip,
                    user_agent=ua,
                    mfa_verified=1 if mfa_verified else 0,
                    revoked=0,
                )
            )
        return session_id

    async def get_session(self, session_id: str) -> Optional[dict]:
        """
        Retrieve a session, returning None if it is expired or revoked.

        Args:
            session_id: The session identifier.

        Returns:
            Session dict, or None if not found / expired / revoked.
        """
        async with self._engine.connect() as conn:
            result = await conn.execute(
                select(user_sessions).where(user_sessions.c.session_id == session_id)
            )
            row = result.fetchone()

        if not row:
            return None

        record = _row_to_dict(row)

        if record.get("revoked"):
            return None

        expires_at = record.get("expires_at")
        if expires_at:
            # Ensure timezone-aware comparison.
            if isinstance(expires_at, datetime):
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
                if _now() > expires_at:
                    return None
        return record

    async def revoke_session(self, session_id: str) -> None:
        """
        Mark a session as revoked.

        Args:
            session_id: The session identifier to revoke.
        """
        async with self._engine.begin() as conn:
            await conn.execute(
                update(user_sessions)
                .where(user_sessions.c.session_id == session_id)
                .values(revoked=1)
            )

    async def revoke_all_sessions(self, user_id: str) -> None:
        """
        Revoke every session belonging to a user (logout everywhere).

        Args:
            user_id: The user whose sessions should all be revoked.
        """
        async with self._engine.begin() as conn:
            await conn.execute(
                update(user_sessions)
                .where(user_sessions.c.user_id == user_id)
                .values(revoked=1)
            )
        logger.info("UserMFAStore: all sessions revoked for user %s", user_id)

    async def get_active_sessions(self, user_id: str) -> list[dict]:
        """
        List all non-revoked, non-expired sessions for a user.

        Args:
            user_id: The user's unique identifier.

        Returns:
            A list of session dicts, sorted by creation time descending.
        """
        now = _now()
        async with self._engine.connect() as conn:
            result = await conn.execute(
                select(user_sessions)
                .where(
                    user_sessions.c.user_id == user_id,
                    user_sessions.c.revoked == 0,
                    user_sessions.c.expires_at > now,
                )
                .order_by(user_sessions.c.created_at.desc())
            )
            rows = result.fetchall()
        return [_row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    async def record_login_attempt(
        self, identifier: str, success: bool, ip: str
    ) -> None:
        """
        Record a login attempt for rate limiting and audit purposes.

        Args:
            identifier: The email address or IP address being tracked.
            success: True if the login succeeded, False if it failed.
            ip: The client's IP address.
        """
        async with self._engine.begin() as conn:
            await conn.execute(
                insert(login_attempts).values(
                    identifier=identifier,
                    success=1 if success else 0,
                    attempted_at=_now(),
                    ip_address=ip,
                )
            )

    async def count_recent_failures(
        self, identifier: str, window_seconds: int = 300
    ) -> int:
        """
        Count failed login attempts for an identifier within a time window.

        Args:
            identifier: The email address or IP address to check.
            window_seconds: How far back to look (default 300s = 5 minutes).

        Returns:
            Number of failed login attempts within the window.
        """
        since = _now() - timedelta(seconds=window_seconds)
        async with self._engine.connect() as conn:
            result = await conn.execute(
                select(func.count())
                .select_from(login_attempts)
                .where(
                    login_attempts.c.identifier == identifier,
                    login_attempts.c.success == 0,
                    login_attempts.c.attempted_at >= since,
                )
            )
            count = result.scalar()
        return count or 0

    async def get_yubikey_identity(self, user_id: str) -> Optional[str]:
        """
        Return the stored YubiKey OTP public identity (12-char modhex) for a user,
        or None if no YubiKey OTP is registered.
        """
        async with self._engine.connect() as conn:
            result = await conn.execute(
                select(user_mfa.c.yubikey_identity).where(
                    user_mfa.c.user_id == user_id
                )
            )
            row = result.fetchone()
        return row[0] if row else None

    async def set_yubikey_identity(
        self, user_id: str, identity: Optional[str]
    ) -> None:
        """
        Store or clear a YubiKey OTP public identity for a user.

        Args:
            user_id: The user to update.
            identity: 12-char modhex identity, or None to remove.
        """
        now = _now()
        async with self._engine.begin() as conn:
            existing = await conn.execute(
                select(user_mfa.c.user_id).where(user_mfa.c.user_id == user_id)
            )
            if existing.fetchone():
                await conn.execute(
                    update(user_mfa)
                    .where(user_mfa.c.user_id == user_id)
                    .values(yubikey_identity=identity, updated_at=now)
                )
            else:
                logger.warning(
                    "set_yubikey_identity: user %s not found in user_mfa", user_id
                )
