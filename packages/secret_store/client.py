"""
Heli.OS Secret Client

Resolves secrets with this priority chain:
  1. Infisical (cloud or self-hosted) — if INFISICAL_TOKEN is set
  2. HashiCorp Vault (KV v2)         — if VAULT_ADDR is set
  3. Environment variable            — development / fallback
  4. Default value                   — non-sensitive config only

This means you can run Heli.OS locally with plain env vars,
move to Infisical (free) for a small team, or Vault for enterprise —
zero code changes required.

Environment variables:
    # Infisical (https://infisical.com — free cloud or self-hosted)
    INFISICAL_TOKEN       - Service token from Infisical project settings
    INFISICAL_PROJECT_ID  - Infisical project ID
    INFISICAL_ENVIRONMENT - "production" | "staging" | "development" (default: production)
    INFISICAL_SITE_URL    - Self-hosted URL (default: https://app.infisical.com)

    # HashiCorp Vault
    VAULT_ADDR          - Vault server URL (e.g. "https://vault.internal:8200")
    VAULT_TOKEN         - Vault token (for token auth)
    VAULT_ROLE_ID       - AppRole RoleID (for AppRole auth)
    VAULT_SECRET_ID     - AppRole SecretID (for AppRole auth)
    VAULT_NAMESPACE     - Vault namespace (Vault Enterprise only)
    VAULT_PATH_PREFIX   - KV path prefix (default: "summit")
    VAULT_MOUNT         - KV v2 mount path (default: "secret")

    SECRET_BACKEND      - "infisical" | "vault" | "env" (default: auto-detect)

Usage:
    from packages.secret_store import get_secret

    db_password = await get_secret("POSTGRES_PASSWORD")
    jwt_secret  = await get_secret("FABRIC_JWT_SECRET", default="dev_secret_only")
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger("summit.secrets")

_VAULT_ADDR = os.getenv("VAULT_ADDR", "")
_VAULT_TOKEN = os.getenv("VAULT_TOKEN", "")
_VAULT_ROLE_ID = os.getenv("VAULT_ROLE_ID", "")
_VAULT_SECRET_ID = os.getenv("VAULT_SECRET_ID", "")
_VAULT_NAMESPACE = os.getenv("VAULT_NAMESPACE", "")
_VAULT_MOUNT = os.getenv("VAULT_MOUNT", "secret")
_VAULT_PATH_PREFIX = os.getenv("VAULT_PATH_PREFIX", "summit")
_SECRET_BACKEND = os.getenv("SECRET_BACKEND", "auto")

_INFISICAL_TOKEN = os.getenv("INFISICAL_TOKEN", "")
_INFISICAL_PROJECT_ID = os.getenv("INFISICAL_PROJECT_ID", "")
_INFISICAL_ENVIRONMENT = os.getenv("INFISICAL_ENVIRONMENT", "production")
_INFISICAL_SITE_URL = os.getenv("INFISICAL_SITE_URL", "https://app.infisical.com")

# Module-level token cache (filled by AppRole auth)
_cached_token: Optional[str] = None


class SecretClient:
    """
    Resolves secrets from Vault (production) or env vars (development).

    Instantiate once per service and reuse.
    """

    def __init__(
        self,
        vault_addr: str = _VAULT_ADDR,
        vault_token: str = _VAULT_TOKEN,
        role_id: str = _VAULT_ROLE_ID,
        secret_id: str = _VAULT_SECRET_ID,
        namespace: str = _VAULT_NAMESPACE,
        mount: str = _VAULT_MOUNT,
        path_prefix: str = _VAULT_PATH_PREFIX,
        backend: str = _SECRET_BACKEND,
        infisical_token: str = _INFISICAL_TOKEN,
        infisical_project_id: str = _INFISICAL_PROJECT_ID,
        infisical_environment: str = _INFISICAL_ENVIRONMENT,
        infisical_site_url: str = _INFISICAL_SITE_URL,
    ):
        self.vault_addr = vault_addr.rstrip("/")
        self.vault_token = vault_token
        self.role_id = role_id
        self.secret_id = secret_id
        self.namespace = namespace
        self.mount = mount
        self.path_prefix = path_prefix
        self.infisical_token = infisical_token
        self.infisical_project_id = infisical_project_id
        self.infisical_environment = infisical_environment
        self.infisical_site_url = infisical_site_url.rstrip("/")
        self._use_infisical = self._should_use_infisical(backend)
        self._use_vault = self._should_use_vault(backend) and not self._use_infisical

        if self._use_infisical:
            logger.info("SecretClient: using Infisical at %s", self.infisical_site_url)
        elif self._use_vault:
            logger.info("SecretClient: using Vault at %s", self.vault_addr)
        else:
            logger.info(
                "SecretClient: using environment variables (no secrets backend configured)"
            )

    def _should_use_infisical(self, backend: str) -> bool:
        if backend == "env" or backend == "vault":
            return False
        if backend == "infisical":
            return True
        # Auto: use Infisical if token + project ID are set
        return bool(self.infisical_token and self.infisical_project_id)

    def _should_use_vault(self, backend: str) -> bool:
        if backend == "env":
            return False
        if backend == "vault":
            return True
        # Auto: use Vault if VAULT_ADDR is set and either token or AppRole creds are present
        return bool(self.vault_addr) and bool(
            self.vault_token or (self.role_id and self.secret_id)
        )

    async def _get_vault_token(self) -> Optional[str]:
        """Get Vault token, using AppRole auth if needed."""
        global _cached_token

        if self.vault_token:
            return self.vault_token

        if _cached_token:
            return _cached_token

        if not (self.role_id and self.secret_id):
            return None

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                headers = {}
                if self.namespace:
                    headers["X-Vault-Namespace"] = self.namespace

                resp = await client.post(
                    f"{self.vault_addr}/v1/auth/approle/login",
                    json={"role_id": self.role_id, "secret_id": self.secret_id},
                    headers=headers,
                )
                resp.raise_for_status()
                _cached_token = resp.json()["auth"]["client_token"]
                logger.info("Vault AppRole auth successful")
                return _cached_token
        except Exception as e:
            logger.error(f"Vault AppRole auth failed: {e}")
            return None

    async def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Resolve a secret by key.

        Args:
            key: Secret key name (same as env var name, e.g. "POSTGRES_PASSWORD")
            default: Fallback if secret not found anywhere

        Returns:
            Secret value, or default if not found
        """
        if self._use_infisical:
            value = await self._get_from_infisical(key)
            if value is not None:
                return value
            logger.debug("Secret '%s' not found in Infisical — falling back to env var", key)

        if self._use_vault:
            value = await self._get_from_vault(key)
            if value is not None:
                return value
            logger.debug("Secret '%s' not found in Vault — falling back to env var", key)

        # Env var fallback
        value = os.getenv(key)
        if value is not None:
            return value

        if default is not None:
            logger.debug(f"Secret '{key}' not found — using default")
            return default

        logger.warning(
            f"Secret '{key}' not found in Vault or environment and no default provided"
        )
        return None

    async def _get_from_infisical(self, key: str) -> Optional[str]:
        """Fetch a secret from Infisical using the REST API (no SDK required)."""
        try:
            headers = {
                "Authorization": f"Bearer {self.infisical_token}",
                "Content-Type": "application/json",
            }
            params = {
                "workspaceId": self.infisical_project_id,
                "environment": self.infisical_environment,
                "secretPath": "/",
                "type": "shared",
            }
            url = f"{self.infisical_site_url}/api/v3/secrets/raw/{key}"
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(url, headers=headers, params=params)
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                return resp.json().get("secret", {}).get("secretValue")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                logger.error("Infisical: unauthorized — check INFISICAL_TOKEN")
            else:
                logger.warning("Infisical error for '%s': %s", key, exc)
            return None
        except Exception as exc:
            logger.warning("Infisical read failed for '%s': %s", key, exc)
            return None

    async def _get_from_vault(self, key: str) -> Optional[str]:
        """Read a secret from Vault KV v2."""
        token = await self._get_vault_token()
        if not token:
            return None

        # Vault KV v2 path: /v1/{mount}/data/{prefix}/{key}
        path = (
            f"{self.vault_addr}/v1/{self.mount}/data/{self.path_prefix}/{key.lower()}"
        )

        try:
            headers = {"X-Vault-Token": token}
            if self.namespace:
                headers["X-Vault-Namespace"] = self.namespace

            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(path, headers=headers)

                if resp.status_code == 404:
                    return None
                resp.raise_for_status()

                data = resp.json().get("data", {}).get("data", {})
                # Look for the key by exact name or lowercased name
                return data.get(key) or data.get(key.lower()) or data.get("value")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                logger.error(
                    f"Vault permission denied for secret '{key}' — check token policies"
                )
            else:
                logger.warning(f"Vault error for '{key}': {e}")
            return None
        except Exception as e:
            logger.warning(f"Vault read failed for '{key}': {e}")
            return None

    async def require(self, key: str) -> str:
        """
        Resolve a secret that must exist.

        Raises ValueError if not found anywhere.
        """
        value = await self.get(key)
        if value is None:
            raise ValueError(
                f"Required secret '{key}' not found. "
                f"Set the env var or add it to Vault at "
                f"{self.vault_addr}/v1/{self.mount}/data/{self.path_prefix}/{key.lower()}"
            )
        return value


# Module-level singleton for convenience
_default_client: Optional[SecretClient] = None


def _get_default_client() -> SecretClient:
    global _default_client
    if _default_client is None:
        _default_client = SecretClient()
    return _default_client


async def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Module-level convenience function.

    from packages.secret_store import get_secret
    password = await get_secret("POSTGRES_PASSWORD")
    """
    return await _get_default_client().get(key, default=default)
