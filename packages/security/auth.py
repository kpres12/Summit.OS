"""
Authentication for Summit.OS

JWT and API Key authentication:
- JWTAuth: issue/verify JWT tokens with configurable claims
- APIKeyAuth: API key management with scoping and rotation
- AuthResult: standardized auth response

Compatible with FastAPI dependency injection.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import base64
import time
import secrets
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum

logger = logging.getLogger("security.auth")


@dataclass
class AuthResult:
    """Result of an authentication attempt."""

    authenticated: bool
    identity: str = ""
    roles: List[str] = field(default_factory=list)
    scopes: List[str] = field(default_factory=list)
    method: str = ""  # "jwt", "api_key", "mtls"
    error: str = ""
    expires_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "authenticated": self.authenticated,
            "identity": self.identity,
            "roles": self.roles,
            "scopes": self.scopes,
            "method": self.method,
            "error": self.error,
        }


# ── JWT Auth ────────────────────────────────────────────────


class JWTAuth:
    """
    JWT token issuer and verifier.

    Uses HMAC-SHA256 for signing. For production, replace with
    RS256 using the mTLS CA key pair.

    Implements:
    - Token issuance with configurable TTL
    - Token verification and claim extraction
    - Token revocation (blacklist)
    - Refresh token support
    """

    def __init__(
        self, secret: str = "", issuer: str = "summit-os", default_ttl: int = 3600
    ):
        self.secret = secret or secrets.token_hex(32)
        self.issuer = issuer
        self.default_ttl = default_ttl
        self._blacklist: Set[str] = set()

    def issue(
        self,
        subject: str,
        roles: Optional[List[str]] = None,
        scopes: Optional[List[str]] = None,
        ttl: Optional[int] = None,
        extra_claims: Optional[Dict] = None,
    ) -> str:
        """Issue a JWT token."""
        now = time.time()
        payload = {
            "sub": subject,
            "iss": self.issuer,
            "iat": int(now),
            "exp": int(now + (ttl or self.default_ttl)),
            "jti": secrets.token_hex(16),
            "roles": roles or [],
            "scopes": scopes or ["read"],
        }
        if extra_claims:
            payload.update(extra_claims)

        return self._encode(payload)

    def verify(self, token: str) -> AuthResult:
        """Verify a JWT token and extract claims."""
        try:
            payload = self._decode(token)
        except Exception as e:
            return AuthResult(authenticated=False, error=str(e), method="jwt")

        # Check expiration
        if payload.get("exp", 0) < time.time():
            return AuthResult(authenticated=False, error="Token expired", method="jwt")

        # Check blacklist
        jti = payload.get("jti", "")
        if jti in self._blacklist:
            return AuthResult(authenticated=False, error="Token revoked", method="jwt")

        # Check issuer
        if payload.get("iss") != self.issuer:
            return AuthResult(authenticated=False, error="Invalid issuer", method="jwt")

        return AuthResult(
            authenticated=True,
            identity=payload.get("sub", ""),
            roles=payload.get("roles", []),
            scopes=payload.get("scopes", []),
            method="jwt",
            expires_at=payload.get("exp", 0),
            metadata=payload,
        )

    def revoke(self, token: str) -> bool:
        """Add token to blacklist."""
        try:
            payload = self._decode(token)
            jti = payload.get("jti")
            if jti:
                self._blacklist.add(jti)
                return True
        except Exception:
            pass
        return False

    def issue_refresh_token(self, subject: str, ttl: int = 86400 * 7) -> str:
        """Issue a long-lived refresh token (7 days default)."""
        return self.issue(subject, scopes=["refresh"], ttl=ttl)

    def refresh(self, refresh_token: str) -> Optional[str]:
        """Exchange a refresh token for a new access token."""
        result = self.verify(refresh_token)
        if not result.authenticated:
            return None
        if "refresh" not in result.scopes:
            return None
        # Revoke old refresh token
        self.revoke(refresh_token)
        # Issue new access token
        return self.issue(result.identity, roles=result.roles)

    def _encode(self, payload: Dict) -> str:
        """Encode payload to JWT."""
        header = {"alg": "HS256", "typ": "JWT"}
        h = self._b64url_encode(json.dumps(header).encode())
        p = self._b64url_encode(json.dumps(payload).encode())
        signature = hmac.new(
            self.secret.encode(), f"{h}.{p}".encode(), hashlib.sha256
        ).digest()
        s = self._b64url_encode(signature)
        return f"{h}.{p}.{s}"

    def _decode(self, token: str) -> Dict:
        """Decode and verify JWT signature."""
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT format")

        h, p, s = parts

        # Verify signature
        expected_sig = hmac.new(
            self.secret.encode(), f"{h}.{p}".encode(), hashlib.sha256
        ).digest()
        actual_sig = self._b64url_decode(s)

        if not hmac.compare_digest(expected_sig, actual_sig):
            raise ValueError("Invalid signature")

        payload = json.loads(self._b64url_decode(p))
        return payload

    @staticmethod
    def _b64url_encode(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    @staticmethod
    def _b64url_decode(s: str) -> bytes:
        padding = 4 - len(s) % 4
        if padding != 4:
            s += "=" * padding
        return base64.urlsafe_b64decode(s)


# ── API Key Auth ────────────────────────────────────────────


@dataclass
class APIKey:
    """An API key with metadata."""

    key_id: str
    key_hash: str  # SHA-256 hash of the actual key
    owner: str
    scopes: List[str] = field(default_factory=lambda: ["read"])
    roles: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0  # 0 = never
    active: bool = True
    description: str = ""


class APIKeyAuth:
    """
    API key management and verification.

    Keys are stored as SHA-256 hashes. The actual key is only
    returned once at creation time.
    """

    def __init__(self):
        self._keys: Dict[str, APIKey] = {}

    def create_key(
        self,
        owner: str,
        scopes: Optional[List[str]] = None,
        roles: Optional[List[str]] = None,
        ttl_days: int = 0,
        description: str = "",
    ) -> Tuple[str, APIKey]:
        """
        Create a new API key.

        Returns (raw_key, APIKey). The raw key is only available at creation.
        """
        raw_key = f"summit_{secrets.token_hex(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key_id = f"key_{secrets.token_hex(8)}"

        expires = 0.0
        if ttl_days > 0:
            expires = time.time() + ttl_days * 86400

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            owner=owner,
            scopes=scopes or ["read"],
            roles=roles or [],
            expires_at=expires,
            description=description,
        )

        self._keys[key_id] = api_key
        logger.info(f"API key created: {key_id} for {owner}")
        return (raw_key, api_key)

    def verify(self, raw_key: str) -> AuthResult:
        """Verify an API key."""
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        for key_id, api_key in self._keys.items():
            if api_key.key_hash == key_hash:
                if not api_key.active:
                    return AuthResult(
                        authenticated=False,
                        error="Key deactivated",
                        method="api_key",
                    )
                if api_key.expires_at > 0 and time.time() > api_key.expires_at:
                    return AuthResult(
                        authenticated=False,
                        error="Key expired",
                        method="api_key",
                    )
                return AuthResult(
                    authenticated=True,
                    identity=api_key.owner,
                    roles=api_key.roles,
                    scopes=api_key.scopes,
                    method="api_key",
                    expires_at=api_key.expires_at,
                )

        return AuthResult(
            authenticated=False, error="Invalid API key", method="api_key"
        )

    def revoke(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self._keys:
            self._keys[key_id].active = False
            logger.info(f"API key revoked: {key_id}")
            return True
        return False

    def rotate(self, key_id: str) -> Optional[Tuple[str, APIKey]]:
        """Rotate an API key (revoke old, create new with same permissions)."""
        old = self._keys.get(key_id)
        if not old:
            return None

        self.revoke(key_id)
        return self.create_key(
            owner=old.owner,
            scopes=old.scopes,
            roles=old.roles,
            description=f"Rotated from {key_id}",
        )

    def list_keys(self, owner: Optional[str] = None) -> List[APIKey]:
        """List API keys, optionally filtered by owner."""
        keys = list(self._keys.values())
        if owner:
            keys = [k for k in keys if k.owner == owner]
        return keys


# For convenient imports
Tuple = tuple  # noqa: avoid shadowing
