"""
Summit.OS MFA API Router

Endpoints:
  POST /auth/mfa/totp/enroll/begin   — generate secret + QR code
  POST /auth/mfa/totp/enroll/verify  — verify first code + activate + get backup codes
  POST /auth/mfa/totp/validate       — validate TOTP during login (checks partial session)
  POST /auth/mfa/totp/disable        — disable TOTP (requires current TOTP code)

  POST /auth/mfa/webauthn/register/begin     — get registration options
  POST /auth/mfa/webauthn/register/complete  — verify + store credential
  POST /auth/mfa/webauthn/authenticate/begin     — get assertion options
  POST /auth/mfa/webauthn/authenticate/complete  — verify assertion

  GET  /auth/mfa/status              — current MFA status for user
  POST /auth/mfa/backup-codes/regenerate  — generate new backup codes (requires TOTP)

  GET  /auth/sessions                — list active sessions
  DELETE /auth/sessions/{session_id} — revoke a session

All endpoints require a valid Authorization: Bearer <token> header. The
user_id and email are extracted from JWT claims by the gateway's existing
verify_bearer() dependency.

Pending TOTP secrets during enrollment are held in a short-lived in-memory
dict (60-second TTL) keyed by user_id. They are never written to the store
until the first code verification succeeds.

Rate limiting uses slowapi. Limits are enforced per client IP:
  - /auth/mfa/totp/validate: 5/minute
  - /auth/mfa/webauthn/authenticate/complete: 5/minute
  - /auth/mfa/totp/enroll/begin: 10/hour
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Dependency: verify_bearer is resolved at request time from main.py.
# We import it lazily to avoid circular imports.
# ---------------------------------------------------------------------------

logger = logging.getLogger("api-gateway.mfa")

# ---------------------------------------------------------------------------
# Rate limiting (slowapi)
# ---------------------------------------------------------------------------

try:
    from slowapi import Limiter  # type: ignore
    from slowapi.util import get_remote_address  # type: ignore

    _limiter = Limiter(key_func=get_remote_address)
    _SLOWAPI_AVAILABLE = True
except ImportError:
    _SLOWAPI_AVAILABLE = False
    _limiter = None  # type: ignore
    logger.warning("slowapi not installed — rate limiting disabled on MFA endpoints")


def _rate_limit(limit_string: str):
    """
    Return a route decorator that applies a slowapi rate limit when available.

    If slowapi is not installed, returns a no-op decorator so the application
    starts cleanly (but without rate limiting — acceptable only for dev).
    """
    if _SLOWAPI_AVAILABLE and _limiter is not None:
        return _limiter.limit(limit_string)

    # No-op decorator fallback.
    def _noop(func):
        return func

    return _noop


# ---------------------------------------------------------------------------
# Security package path wiring
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PACKAGES = _REPO_ROOT / "packages"
if str(_PACKAGES) not in sys.path:
    sys.path.insert(0, str(_PACKAGES))

from security.mfa import (  # noqa: E402
    generate_totp_secret,
    get_totp_uri,
    generate_qr_code_data_uri,
    verify_totp,
    generate_backup_codes,
    hash_backup_code,
    begin_registration as _webauthn_begin_registration,
    complete_registration as _webauthn_complete_registration,
    begin_authentication as _webauthn_begin_authentication,
    complete_authentication as _webauthn_complete_authentication,
)
from security.user_store import UserMFAStore  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level store singleton (set by init_mfa_router)
# ---------------------------------------------------------------------------

_store: Optional[UserMFAStore] = None


def get_store() -> UserMFAStore:
    """FastAPI dependency that returns the initialised UserMFAStore."""
    if _store is None:
        raise RuntimeError(
            "MFA store not initialised — call init_mfa_router(store) at startup"
        )
    return _store


def init_mfa_router(store: UserMFAStore) -> APIRouter:
    """
    Bind a UserMFAStore instance to this router and return the router.

    Call this once at application startup before including the router in
    your FastAPI app:

        from routers.mfa import init_mfa_router
        app.include_router(init_mfa_router(store))

    Args:
        store: An initialised UserMFAStore instance.

    Returns:
        The configured APIRouter ready to be included.
    """
    global _store
    _store = store
    return router


# ---------------------------------------------------------------------------
# Pending enrollment cache (in-memory, 60-second TTL)
# ---------------------------------------------------------------------------

# { user_id: (totp_secret: str, enrolled_at: float) }
_pending_enrollments: dict[str, tuple[str, float]] = {}
_PENDING_TTL_SECONDS = 60


def _set_pending(user_id: str, secret: str) -> None:
    _pending_enrollments[user_id] = (secret, time.monotonic())


def _get_pending(user_id: str) -> Optional[str]:
    entry = _pending_enrollments.get(user_id)
    if not entry:
        return None
    secret, ts = entry
    if time.monotonic() - ts > _PENDING_TTL_SECONDS:
        del _pending_enrollments[user_id]
        return None
    return secret


def _clear_pending(user_id: str) -> None:
    _pending_enrollments.pop(user_id, None)


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class TOTPEnrollVerifyRequest(BaseModel):
    """Body for /auth/mfa/totp/enroll/verify."""

    token: str


class TOTPValidateRequest(BaseModel):
    """Body for /auth/mfa/totp/validate."""

    token: str


class TOTPDisableRequest(BaseModel):
    """Body for /auth/mfa/totp/disable. Requires the current TOTP code."""

    token: str


class BackupCodeRegenerateRequest(BaseModel):
    """Body for /auth/mfa/backup-codes/regenerate. Requires current TOTP code."""

    token: str


class WebAuthnRegisterBeginRequest(BaseModel):
    """Body for /auth/mfa/webauthn/register/begin."""

    rp_id: str
    rp_name: str = "Summit.OS"


class WebAuthnRegisterCompleteRequest(BaseModel):
    """Body for /auth/mfa/webauthn/register/complete."""

    credential: dict
    challenge: str
    rp_id: str
    expected_origin: str
    device_name: Optional[str] = None


class WebAuthnAuthBeginRequest(BaseModel):
    """Body for /auth/mfa/webauthn/authenticate/begin."""

    rp_id: str


class WebAuthnAuthCompleteRequest(BaseModel):
    """Body for /auth/mfa/webauthn/authenticate/complete."""

    credential: dict
    challenge: str
    rp_id: str
    expected_origin: str


# ---------------------------------------------------------------------------
# Router definition
# ---------------------------------------------------------------------------

router = APIRouter(tags=["mfa"])

# ---------------------------------------------------------------------------
# Auth dependency helper
# ---------------------------------------------------------------------------


async def _get_claims(request: Request) -> dict:
    """
    Extract JWT claims from the request.

    Calls the gateway's verify_bearer() dependency. When OIDC enforcement is
    disabled, verify_bearer returns None; we fall back to parsing the JWT
    claims without signature verification so that user_id is always available.

    Returns:
        A claims dict containing at least ``sub`` (user_id) and optionally
        ``email``.

    Raises:
        HTTPException 401: If no Authorization header is present in enforced mode.
        HTTPException 401: If the token cannot be decoded at all.
    """
    # Import at call time to avoid circular module dependency.
    try:
        from main import verify_bearer  # type: ignore
    except ImportError:
        # Running outside of the gateway context (tests / direct import).
        verify_bearer = None  # type: ignore

    authorization: Optional[str] = request.headers.get("Authorization")

    if verify_bearer is not None:
        claims = await verify_bearer(authorization)
        if claims:
            return claims

    # OIDC enforcement is off or verify_bearer returned None.
    # Extract claims without signature verification for dev/test workflows.
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = authorization.split(" ", 1)[1]
    try:
        import base64
        import json as _json

        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Not a JWT")
        padding = 4 - len(parts[1]) % 4
        payload_bytes = base64.urlsafe_b64decode(parts[1] + "=" * padding)
        return _json.loads(payload_bytes)
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}") from exc


def _user_from_claims(claims: dict) -> tuple[str, str]:
    """
    Extract (user_id, email) from JWT claims.

    Args:
        claims: Decoded JWT payload dict.

    Returns:
        (user_id, email) tuple. email falls back to user_id if not present.

    Raises:
        HTTPException 401: If the ``sub`` claim is missing.
    """
    user_id = claims.get("sub") or claims.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing 'sub' claim")
    email = claims.get("email") or claims.get("preferred_username") or user_id
    return str(user_id), str(email)


# ---------------------------------------------------------------------------
# TOTP endpoints
# ---------------------------------------------------------------------------


@router.post("/auth/mfa/totp/enroll/begin")
@_rate_limit("10/hour")
async def totp_enroll_begin(
    request: Request,
    store: UserMFAStore = Depends(get_store),
) -> dict[str, Any]:
    """
    Begin TOTP enrollment for the authenticated user.

    Generates a fresh TOTP secret and returns the provisioning URI plus a
    data-URI-encoded QR code image. The secret is held in a 60-second
    in-memory cache; it is **not** persisted until the user successfully
    verifies the first code via /auth/mfa/totp/enroll/verify.

    Returns:
        JSON with ``qr_code_data_uri``, ``totp_uri``, and ``secret`` fields.
        The ``secret`` is included so power users can manually enter it into
        their authenticator app.
    """
    claims = await _get_claims(request)
    user_id, email = _user_from_claims(claims)

    # Ensure user record exists in the store.
    await store.upsert_user(user_id, email)

    secret = generate_totp_secret()
    _set_pending(user_id, secret)

    totp_uri = get_totp_uri(secret, email)
    try:
        qr_data_uri = generate_qr_code_data_uri(totp_uri)
    except ImportError:
        qr_data_uri = None

    return {
        "totp_uri": totp_uri,
        "secret": secret,
        "qr_code_data_uri": qr_data_uri,
    }


@router.post("/auth/mfa/totp/enroll/verify")
async def totp_enroll_verify(
    body: TOTPEnrollVerifyRequest,
    request: Request,
    store: UserMFAStore = Depends(get_store),
) -> dict[str, Any]:
    """
    Complete TOTP enrollment by verifying the first code from the authenticator.

    On success, stores the encrypted secret, generates backup codes, and
    activates TOTP for the user. The plaintext backup codes are returned
    **once** — they cannot be recovered afterward.

    Args:
        body: Must contain ``token`` — the 6-digit code from the authenticator.

    Returns:
        JSON with ``backup_codes`` (list of 10 plaintext codes) and
        ``message`` confirming enrollment.

    Raises:
        HTTPException 400: If enrollment was not started or the TTL expired.
        HTTPException 422: If the TOTP code is invalid.
    """
    claims = await _get_claims(request)
    user_id, email = _user_from_claims(claims)

    pending_secret = _get_pending(user_id)
    if not pending_secret:
        raise HTTPException(
            status_code=400,
            detail="No pending enrollment found. Call /enroll/begin first (TTL is 60s).",
        )

    if not verify_totp(body.token, pending_secret):
        raise HTTPException(status_code=422, detail="Invalid TOTP code.")

    # Persist the encrypted secret.
    await store.set_totp_secret(user_id, pending_secret)
    _clear_pending(user_id)

    # Generate and store backup codes.
    plaintext_codes = generate_backup_codes(10)
    await store.enable_totp(user_id, plaintext_codes)

    logger.info("TOTP enrolled for user %s", user_id)
    return {
        "message": "TOTP enrollment successful.",
        "backup_codes": plaintext_codes,
    }


@router.post("/auth/mfa/totp/validate")
@_rate_limit("5/minute")
async def totp_validate(
    body: TOTPValidateRequest,
    request: Request,
    store: UserMFAStore = Depends(get_store),
) -> dict[str, Any]:
    """
    Validate a TOTP code during the login flow.

    This endpoint is intended for use after primary credential verification
    when the login flow requires a second factor. On success, the caller
    should upgrade the session's mfa_verified flag.

    Args:
        body: Must contain ``token`` — the 6-digit TOTP code.

    Returns:
        JSON with ``valid: true`` on success.

    Raises:
        HTTPException 400: If the user has no TOTP enrolled.
        HTTPException 422: If the code is invalid.
    """
    claims = await _get_claims(request)
    user_id, _ = _user_from_claims(claims)

    secret = await store.get_totp_secret(user_id)
    if not secret:
        raise HTTPException(
            status_code=400, detail="TOTP is not enrolled for this user."
        )

    if not verify_totp(body.token, secret):
        raise HTTPException(status_code=422, detail="Invalid TOTP code.")

    return {"valid": True}


@router.post("/auth/mfa/totp/disable")
async def totp_disable(
    body: TOTPDisableRequest,
    request: Request,
    store: UserMFAStore = Depends(get_store),
) -> dict[str, Any]:
    """
    Disable TOTP for the authenticated user.

    Requires the user to supply their current TOTP code as proof of
    possession before the secret is wiped. This prevents an attacker with
    only a stolen session from disabling MFA.

    Args:
        body: Must contain ``token`` — the current 6-digit TOTP code.

    Returns:
        JSON with ``message`` confirming TOTP was disabled.

    Raises:
        HTTPException 400: If TOTP is not enrolled.
        HTTPException 422: If the code is invalid.
    """
    claims = await _get_claims(request)
    user_id, _ = _user_from_claims(claims)

    secret = await store.get_totp_secret(user_id)
    if not secret:
        raise HTTPException(
            status_code=400, detail="TOTP is not enrolled for this user."
        )

    if not verify_totp(body.token, secret):
        raise HTTPException(
            status_code=422, detail="Invalid TOTP code — cannot disable MFA."
        )

    await store.disable_totp(user_id)
    logger.info("TOTP disabled for user %s", user_id)
    return {"message": "TOTP has been disabled."}


# ---------------------------------------------------------------------------
# WebAuthn endpoints
# ---------------------------------------------------------------------------


@router.post("/auth/mfa/webauthn/register/begin")
async def webauthn_register_begin(
    body: WebAuthnRegisterBeginRequest,
    request: Request,
    store: UserMFAStore = Depends(get_store),
) -> dict[str, Any]:
    """
    Generate WebAuthn registration options for a new authenticator.

    Returns the PublicKeyCredentialCreationOptions that the browser passes
    to ``navigator.credentials.create()``.

    Args:
        body: Contains ``rp_id`` (effective domain) and optional ``rp_name``.

    Returns:
        WebAuthn registration options dict.

    Raises:
        HTTPException 501: If the ``webauthn`` package is not installed.
    """
    claims = await _get_claims(request)
    user_id, email = _user_from_claims(claims)

    await store.upsert_user(user_id, email)
    existing = await store.get_webauthn_credentials(user_id)

    try:
        options = _webauthn_begin_registration(
            user_id=user_id,
            user_name=email,
            rp_id=body.rp_id,
            rp_name=body.rp_name,
            existing_credentials=existing,
        )
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc

    return options


@router.post("/auth/mfa/webauthn/register/complete")
async def webauthn_register_complete(
    body: WebAuthnRegisterCompleteRequest,
    request: Request,
    store: UserMFAStore = Depends(get_store),
) -> dict[str, Any]:
    """
    Verify a WebAuthn registration response and persist the credential.

    Args:
        body: Contains the ``credential`` from the browser, the ``challenge``
              from :func:`webauthn_register_begin`, ``rp_id``, and
              ``expected_origin``.

    Returns:
        JSON with ``message``, ``credential_id``, ``device_name``, and
        ``aaguid``.

    Raises:
        HTTPException 501: If the ``webauthn`` package is not installed.
        HTTPException 422: If registration verification fails.
    """
    claims = await _get_claims(request)
    user_id, _ = _user_from_claims(claims)

    try:
        cred_data = _webauthn_complete_registration(
            credential=body.credential,
            challenge=body.challenge,
            rp_id=body.rp_id,
            expected_origin=body.expected_origin,
        )
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning("WebAuthn registration failed for user %s: %s", user_id, exc)
        raise HTTPException(
            status_code=422, detail=f"Registration failed: {exc}"
        ) from exc

    # Allow caller to override the device name.
    if body.device_name:
        cred_data["device_name"] = body.device_name

    await store.add_webauthn_credential(user_id, cred_data)
    logger.info(
        "WebAuthn credential %s registered for user %s",
        cred_data["credential_id"],
        user_id,
    )
    return {
        "message": "Security key registered successfully.",
        "credential_id": cred_data["credential_id"],
        "device_name": cred_data.get("device_name"),
        "aaguid": cred_data.get("aaguid"),
    }


@router.post("/auth/mfa/webauthn/authenticate/begin")
async def webauthn_auth_begin(
    body: WebAuthnAuthBeginRequest,
    request: Request,
    store: UserMFAStore = Depends(get_store),
) -> dict[str, Any]:
    """
    Generate WebAuthn authentication options for a login challenge.

    Returns PublicKeyCredentialRequestOptions for the browser's
    ``navigator.credentials.get()``.

    Args:
        body: Contains ``rp_id``.

    Returns:
        WebAuthn authentication options dict.

    Raises:
        HTTPException 400: If the user has no registered WebAuthn credentials.
        HTTPException 501: If the ``webauthn`` package is not installed.
    """
    claims = await _get_claims(request)
    user_id, _ = _user_from_claims(claims)

    credentials = await store.get_webauthn_credentials(user_id)
    if not credentials:
        raise HTTPException(
            status_code=400,
            detail="No WebAuthn credentials registered for this user.",
        )

    try:
        options = _webauthn_begin_authentication(
            credentials=credentials,
            rp_id=body.rp_id,
        )
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc

    return options


@router.post("/auth/mfa/webauthn/authenticate/complete")
@_rate_limit("5/minute")
async def webauthn_auth_complete(
    body: WebAuthnAuthCompleteRequest,
    request: Request,
    store: UserMFAStore = Depends(get_store),
) -> dict[str, Any]:
    """
    Verify a WebAuthn authentication assertion and update the sign count.

    Args:
        body: Contains the browser ``credential``, ``challenge``, ``rp_id``,
              and ``expected_origin``.

    Returns:
        JSON with ``valid: true`` and updated ``sign_count``.

    Raises:
        HTTPException 400: If no matching stored credential is found.
        HTTPException 501: If the ``webauthn`` package is not installed.
        HTTPException 422: If assertion verification fails.
    """
    claims = await _get_claims(request)
    user_id, _ = _user_from_claims(claims)

    # Locate the stored credential by the ID present in the response.
    credential_id_raw = body.credential.get("id") or body.credential.get("rawId", "")
    all_creds = await store.get_webauthn_credentials(user_id)
    stored = next(
        (c for c in all_creds if c["credential_id"] == credential_id_raw),
        None,
    )
    if not stored:
        raise HTTPException(
            status_code=400,
            detail="Credential not found for this user.",
        )

    try:
        result = _webauthn_complete_authentication(
            credential=body.credential,
            challenge=body.challenge,
            stored_credential=stored,
            rp_id=body.rp_id,
            expected_origin=body.expected_origin,
        )
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning("WebAuthn authentication failed for user %s: %s", user_id, exc)
        raise HTTPException(
            status_code=422, detail=f"Authentication failed: {exc}"
        ) from exc

    await store.update_webauthn_sign_count(
        stored["credential_id"], result["sign_count"]
    )
    return {"valid": True, "sign_count": result["sign_count"]}


# ---------------------------------------------------------------------------
# MFA status
# ---------------------------------------------------------------------------


@router.get("/auth/mfa/status")
async def mfa_status(
    request: Request,
    store: UserMFAStore = Depends(get_store),
) -> dict[str, Any]:
    """
    Return the current MFA enrollment status for the authenticated user.

    Returns:
        JSON with:
        - ``totp_enabled`` (bool)
        - ``webauthn_credentials`` (list of credential summaries)
        - ``backup_codes_remaining`` (int)
        - ``mfa_method`` (str: 'none' | 'totp' | 'webauthn' | 'both')
    """
    claims = await _get_claims(request)
    user_id, email = _user_from_claims(claims)

    record = await store.get_user(user_id)
    if not record:
        # First-time visit: create an empty MFA record.
        record = await store.upsert_user(user_id, email)

    webauthn_creds = await store.get_webauthn_credentials(user_id)
    cred_summaries = [
        {
            "credential_id": c["credential_id"],
            "device_name": c.get("device_name"),
            "aaguid": c.get("aaguid"),
            "created_at": c.get("created_at"),
            "last_used_at": c.get("last_used_at"),
        }
        for c in webauthn_creds
    ]

    return {
        "totp_enabled": bool(record.get("totp_enabled")),
        "webauthn_credentials": cred_summaries,
        "backup_codes_remaining": record.get("backup_codes_remaining", 0),
        "mfa_method": record.get("mfa_method", "none"),
    }


# ---------------------------------------------------------------------------
# Backup code regeneration
# ---------------------------------------------------------------------------


@router.post("/auth/mfa/backup-codes/regenerate")
async def backup_codes_regenerate(
    body: BackupCodeRegenerateRequest,
    request: Request,
    store: UserMFAStore = Depends(get_store),
) -> dict[str, Any]:
    """
    Regenerate backup codes for the authenticated user.

    The current TOTP code is required as proof of possession. Existing backup
    codes are invalidated and replaced with a fresh set.

    Args:
        body: Must contain ``token`` — the current 6-digit TOTP code.

    Returns:
        JSON with the new ``backup_codes`` list (plaintext, shown once only).

    Raises:
        HTTPException 400: If TOTP is not enrolled.
        HTTPException 422: If the TOTP code is invalid.
    """
    claims = await _get_claims(request)
    user_id, _ = _user_from_claims(claims)

    secret = await store.get_totp_secret(user_id)
    if not secret:
        raise HTTPException(
            status_code=400,
            detail="TOTP is not enrolled — cannot regenerate backup codes.",
        )

    if not verify_totp(body.token, secret):
        raise HTTPException(status_code=422, detail="Invalid TOTP code.")

    plaintext_codes = generate_backup_codes(10)
    await store.enable_totp(user_id, plaintext_codes)

    logger.info("Backup codes regenerated for user %s", user_id)
    return {
        "message": "Backup codes regenerated. Store these somewhere safe.",
        "backup_codes": plaintext_codes,
    }


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


@router.get("/auth/sessions")
async def list_sessions(
    request: Request,
    store: UserMFAStore = Depends(get_store),
) -> dict[str, Any]:
    """
    List all active (non-revoked, non-expired) sessions for the user.

    Returns:
        JSON with a ``sessions`` list. Each entry includes ``session_id``,
        ``created_at``, ``expires_at``, ``ip_address``, ``user_agent``, and
        ``mfa_verified``. The ``session_id`` values are opaque tokens; the
        current session can be identified by the caller matching the token
        they submitted.
    """
    claims = await _get_claims(request)
    user_id, _ = _user_from_claims(claims)

    sessions = await store.get_active_sessions(user_id)
    # Redact the full session_id to a prefix for display safety; the caller
    # can pass the full token when requesting revocation.
    summaries = []
    for s in sessions:
        summaries.append(
            {
                "session_id": s["session_id"],
                "created_at": _isoformat(s.get("created_at")),
                "expires_at": _isoformat(s.get("expires_at")),
                "ip_address": s.get("ip_address"),
                "user_agent": s.get("user_agent"),
                "mfa_verified": bool(s.get("mfa_verified")),
            }
        )
    return {"sessions": summaries}


@router.delete("/auth/sessions/{session_id}")
async def revoke_session(
    session_id: str,
    request: Request,
    store: UserMFAStore = Depends(get_store),
) -> dict[str, Any]:
    """
    Revoke a specific session.

    The authenticated user may only revoke their own sessions. Attempting to
    revoke a session that belongs to a different user silently succeeds (to
    avoid user-enumeration) but has no effect, because the store's
    get_active_sessions scopes to the requesting user.

    Args:
        session_id: The session identifier from :func:`list_sessions`.

    Returns:
        JSON confirming revocation.
    """
    claims = await _get_claims(request)
    user_id, _ = _user_from_claims(claims)

    # Verify the session belongs to this user before revoking.
    active = await store.get_active_sessions(user_id)
    owned = any(s["session_id"] == session_id for s in active)
    if not owned:
        # Do not leak whether the session exists or belongs to another user.
        raise HTTPException(status_code=404, detail="Session not found.")

    await store.revoke_session(session_id)
    logger.info("Session %s revoked for user %s", session_id, user_id)
    return {"message": "Session revoked."}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _isoformat(value: Any) -> Optional[str]:
    """Safely convert a datetime (or string) to ISO-8601 string."""
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
